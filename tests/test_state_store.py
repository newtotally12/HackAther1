import pytest

from ares.state_store import StateStore


@pytest.mark.asyncio
async def test_local_encrypted_state_persistence(tmp_path) -> None:
    path = tmp_path / "state.enc"
    first = StateStore(
        bot_token="test-token",
        admin_id=123,
        repository=None,
        github_token=None,
        branch="main",
        local_path=str(path),
    )
    state = await first.load()
    state["providers"] = {"one": {"name": "private provider", "api_key": "top-secret"}}
    await first.save(state)
    await first.close()

    second = StateStore(
        bot_token="test-token",
        admin_id=123,
        repository=None,
        github_token=None,
        branch="main",
        local_path=str(path),
    )
    loaded = await second.load()
    assert loaded["providers"]["one"]["api_key"] == "top-secret"
    assert b"top-secret" not in path.read_bytes()
    await second.close()


@pytest.mark.asyncio
async def test_github_binary_state_falls_back_to_git_blob(tmp_path) -> None:
    from base64 import b64encode

    import httpx

    from ares.crypto import encrypt_state

    bot_token = "test-token"
    repository = "owner/repo"
    admin_id = 123
    expected = {"providers": {"one": {"api_key": "preserved-secret"}}}
    encrypted = encrypt_state(
        expected,
        bot_token=bot_token,
        repository=repository,
        admin_id=admin_id,
    )
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(request.url.path)
        if request.url.path.endswith("/contents/.ares/state.enc"):
            return httpx.Response(
                200,
                json={"sha": "state-blob-sha", "encoding": "none", "content": "not base64!"},
            )
        if request.url.path.endswith("/git/blobs/state-blob-sha"):
            return httpx.Response(
                200,
                json={
                    "sha": "state-blob-sha",
                    "encoding": "base64",
                    "content": b64encode(encrypted).decode("ascii"),
                },
            )
        return httpx.Response(404)

    store = StateStore(
        bot_token=bot_token,
        admin_id=admin_id,
        repository=repository,
        github_token="test-github-token",
        branch="main",
        local_path=str(tmp_path / "state.enc"),
    )
    await store._client.aclose()
    store._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    try:
        loaded = await store.load()
        assert loaded["providers"]["one"]["api_key"] == "preserved-secret"
        assert store.remote_sha == "state-blob-sha"
        assert any(path.endswith("/git/blobs/state-blob-sha") for path in requested)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_github_state_save_refreshes_sha_and_retries_conflict(tmp_path) -> None:
    import httpx

    store = StateStore(
        bot_token="test-token",
        admin_id=123,
        repository="owner/repo",
        github_token="test-github-token",
        branch="main",
        local_path=str(tmp_path / "state.enc"),
    )
    await store._client.aclose()
    puts = 0
    refreshed = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal puts, refreshed
        if request.method == "PUT":
            puts += 1
            if puts == 1:
                return httpx.Response(409, json={"message": "Conflict"})
            assert refreshed is True
            assert request.read()  # The retried write still carries the encrypted state.
            import json

            body = json.loads(request.content)
            assert body["sha"] == "fresh-remote-sha"
            return httpx.Response(200, json={"content": {"sha": "updated-remote-sha"}})
        if request.method == "GET" and request.url.path.endswith("/contents/.ares/state.enc"):
            refreshed = True
            return httpx.Response(200, json={"sha": "fresh-remote-sha"})
        return httpx.Response(404)

    store._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        await store.save({"active_run": "scan-1"})
        assert puts == 2
        assert store.remote_sha == "updated-remote-sha"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_concurrent_state_saves_are_serialized(tmp_path, monkeypatch) -> None:
    import asyncio
    import httpx

    store = StateStore(
        bot_token="test-token",
        admin_id=123,
        repository="owner/repo",
        github_token="test-github-token",
        branch="main",
        local_path=str(tmp_path / "state.enc"),
    )
    active_puts = 0
    maximum_active_puts = 0
    completed_puts = 0

    async def fake_put(*_args, **_kwargs) -> httpx.Response:
        nonlocal active_puts, maximum_active_puts, completed_puts
        active_puts += 1
        maximum_active_puts = max(maximum_active_puts, active_puts)
        await asyncio.sleep(0.01)
        completed_puts += 1
        active_puts -= 1
        request = httpx.Request("PUT", "https://api.github.com/state")
        return httpx.Response(
            200,
            json={"content": {"sha": f"remote-sha-{completed_puts}"}},
            request=request,
        )

    monkeypatch.setattr(store._client, "put", fake_put)
    try:
        await asyncio.gather(
            store.save({"active_run": "scan-1"}),
            store.save({"active_run": "scan-2"}),
        )
        assert maximum_active_puts == 1
        assert completed_puts == 2
        assert store.remote_sha == "remote-sha-2"
    finally:
        await store.close()

@pytest.mark.asyncio
async def test_state_from_another_repository_starts_fresh_and_is_replaced(tmp_path) -> None:
    import json
    from base64 import b64decode, b64encode

    import httpx

    from ares.crypto import decrypt_state, encrypt_state
    from ares.state_store import DEFAULT_STATE

    current_token = "fork-bot-token"
    current_repository = "newtotally12/HackAther1"
    admin_id = 123
    inherited = encrypt_state(
        {"providers": {"legacy": {"api_key": "must-not-carry-over"}}},
        bot_token=current_token,
        repository="Aryasamadi/HackAther",
        admin_id=admin_id,
    )
    saved_payloads: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path.endswith("/contents/.ares/state.enc"):
            return httpx.Response(
                200,
                json={
                    "sha": "inherited-state-sha",
                    "encoding": "base64",
                    "content": b64encode(inherited).decode("ascii"),
                },
            )
        if request.method == "PUT" and request.url.path.endswith("/contents/.ares/state.enc"):
            body = json.loads(request.content)
            assert body["sha"] == "inherited-state-sha"
            fresh_payload = b64decode(body["content"])
            assert decrypt_state(
                fresh_payload,
                bot_token=current_token,
                repository=current_repository,
                admin_id=admin_id,
            ) == DEFAULT_STATE
            saved_payloads.append(fresh_payload)
            return httpx.Response(200, json={"content": {"sha": "fresh-state-sha"}})
        return httpx.Response(404)

    store = StateStore(
        bot_token=current_token,
        admin_id=admin_id,
        repository=current_repository,
        github_token="test-github-token",
        branch="main",
        local_path=str(tmp_path / "state.enc"),
    )
    await store._client.aclose()
    store._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        loaded = await store.load()
        assert loaded == DEFAULT_STATE
        assert store.recovered_from_invalid_key is True
        assert store.remote_sha == "inherited-state-sha"

        await store.save(loaded)
        assert len(saved_payloads) == 1
        assert store.remote_sha == "fresh-state-sha"
    finally:
        await store.close()
