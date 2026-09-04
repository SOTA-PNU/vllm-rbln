#!/usr/bin/env python3
"""Minimal OpenAI-compatible proxy for prefill/decode disaggregation."""

# Copyright 2025 Rebellions Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse


def _port(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Route an OpenAI completion through prefill and decode servers."
    )
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=_port)
    parser.add_argument("--prefill-host", "--prefiller-host", required=True)
    parser.add_argument("--prefill-port", "--prefiller-port", required=True, type=_port)
    parser.add_argument("--decode-host", "--decoder-host", required=True)
    parser.add_argument("--decode-port", "--decoder-port", required=True, type=_port)
    return parser.parse_args()


def _base_url(host: str, port: int) -> str:
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{port}"


def _request_headers(request: Request, request_id: str) -> dict[str, str]:
    headers = {"x-request-id": request_id}
    authorization = request.headers.get("authorization")
    if authorization:
        headers["authorization"] = authorization
    elif api_key := os.environ.get("OPENAI_API_KEY"):
        headers["authorization"] = f"Bearer {api_key}"
    return headers


def _response_headers(response: httpx.Response, request_id: str) -> dict[str, str]:
    # httpx decodes content encodings, and the ASGI server owns hop-by-hop headers.
    excluded = {
        "connection",
        "content-encoding",
        "content-length",
        "date",
        "server",
        "transfer-encoding",
    }
    headers = {
        name: value
        for name, value in response.headers.items()
        if name.lower() not in excluded
    }
    headers["x-request-id"] = response.headers.get("x-request-id", request_id)
    return headers


def _prefill_payload(payload: dict[str, object]) -> dict[str, object]:
    prefill_payload = payload.copy()
    prefill_payload["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }
    prefill_payload["stream"] = False
    prefill_payload["max_tokens"] = 1
    if "max_completion_tokens" in prefill_payload:
        prefill_payload["max_completion_tokens"] = 1
    prefill_payload.pop("stream_options", None)
    return prefill_payload


def _forward_response(response: httpx.Response, request_id: str) -> Response:
    return Response(
        content=response.content,
        status_code=response.status_code,
        headers=_response_headers(response, request_id),
    )


async def _stream_body(response: httpx.Response) -> AsyncIterator[bytes]:
    try:
        async for chunk in response.aiter_bytes():
            yield chunk
    finally:
        await response.aclose()


def create_app(args: argparse.Namespace) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        limits = httpx.Limits(
            max_connections=None,
            max_keepalive_connections=None,
        )
        app.state.prefill_client = httpx.AsyncClient(
            base_url=_base_url(args.prefill_host, args.prefill_port),
            timeout=None,
            limits=limits,
        )
        app.state.decode_client = httpx.AsyncClient(
            base_url=_base_url(args.decode_host, args.decode_port),
            timeout=None,
            limits=limits,
        )
        try:
            yield
        finally:
            await app.state.prefill_client.aclose()
            await app.state.decode_client.aclose()

    app = FastAPI(lifespan=lifespan)

    @app.get("/healthcheck")
    async def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/completions")
    async def completions(request: Request) -> Response:
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        try:
            payload = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="invalid JSON request") from exc
        if not isinstance(payload, dict):
            raise HTTPException(
                status_code=400,
                detail="request JSON must be an object",
            )

        headers = _request_headers(request, request_id)
        try:
            prefill_response = await request.app.state.prefill_client.post(
                "/v1/completions",
                json=_prefill_payload(payload),
                headers=headers,
            )
        except httpx.RequestError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"prefill request failed: {exc}",
            ) from exc

        if prefill_response.is_error:
            return _forward_response(prefill_response, request_id)
        try:
            prefill_response_payload = prefill_response.json()
        except ValueError as exc:
            raise HTTPException(
                status_code=502,
                detail="prefill returned a non-JSON response",
            ) from exc
        if not isinstance(prefill_response_payload, dict):
            raise HTTPException(
                status_code=502,
                detail="prefill response JSON must be an object",
            )
        transfer_params = prefill_response_payload.get("kv_transfer_params")
        if not isinstance(transfer_params, dict) or not transfer_params:
            raise HTTPException(
                status_code=502,
                detail="prefill response omitted kv_transfer_params",
            )
        required_transfer_params = {
            "do_remote_prefill",
            "remote_block_ids",
            "remote_engine_id",
            "remote_request_id",
            "remote_host",
            "remote_port",
            "tp_size",
        }
        missing_transfer_params = required_transfer_params - transfer_params.keys()
        if missing_transfer_params:
            missing = ", ".join(sorted(missing_transfer_params))
            raise HTTPException(
                status_code=502,
                detail=f"prefill response omitted transfer fields: {missing}",
            )
        if transfer_params["do_remote_prefill"] is not True:
            raise HTTPException(
                status_code=502,
                detail="prefill response did not request remote prefill",
            )

        decode_payload = payload.copy()
        decode_payload["kv_transfer_params"] = transfer_params
        try:
            decode_request = request.app.state.decode_client.build_request(
                "POST",
                "/v1/completions",
                json=decode_payload,
                headers=headers,
            )
            decode_response = await request.app.state.decode_client.send(
                decode_request,
                stream=bool(payload.get("stream", False)),
            )
        except httpx.RequestError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"decode request failed: {exc}",
            ) from exc

        if not payload.get("stream", False):
            await decode_response.aread()
            await decode_response.aclose()
            return _forward_response(decode_response, request_id)
        if decode_response.is_error:
            await decode_response.aread()
            await decode_response.aclose()
            return _forward_response(decode_response, request_id)
        return StreamingResponse(
            _stream_body(decode_response),
            status_code=decode_response.status_code,
            headers=_response_headers(decode_response, request_id),
        )

    return app


def main() -> None:
    args = parse_args()
    uvicorn.run(create_app(args), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
