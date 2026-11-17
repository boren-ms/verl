# Copyright 2026 Individual Contributor: Boren
# SPDX-License-Identifier: Apache-2.0
"""Check that TQ isolation changes only its own namespace lookups."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from verl.utils import transferqueue_namespace as module


def test_scope_and_idempotence(monkeypatch):
    ray = SimpleNamespace(get_actor=Mock(), get=Mock())
    controller = SimpleNamespace(options=Mock())
    storage_actor = SimpleNamespace(options=Mock())
    interface = SimpleNamespace(ray=ray, TransferQueueController=controller)
    storage = SimpleNamespace(ray=ray, RayObjectRefStorage=storage_actor)
    bootstrap = SimpleNamespace(SimpleStorageUnit=SimpleNamespace(options=Mock()))
    modules = {
        "transfer_queue.interface": interface,
        "transfer_queue.storage.clients.ray_storage_client": storage,
        "transfer_queue.storage.bootstrap.simple_storage_bootstrap": bootstrap,
    }
    monkeypatch.setattr(module.importlib, "import_module", modules.__getitem__)
    monkeypatch.setenv("VERL_TRANSFER_QUEUE_NAMESPACE", "evaluation-20")
    module.configure_transfer_queue_namespace()
    proxy = interface.ray
    module.configure_transfer_queue_namespace()
    assert interface.ray is proxy
    interface.ray.get_actor("TransferQueueController", namespace="transfer_queue")
    ray.get_actor.assert_called_with("TransferQueueController", namespace="evaluation-20")
    interface.TransferQueueController.options(name="TransferQueueController", namespace="transfer_queue")
    controller.options.assert_called_with(name="TransferQueueController", namespace="evaluation-20")
    storage.RayObjectRefStorage.options(namespace="transfer_queue")
    storage_actor.options.assert_called_with(namespace="evaluation-20")
    storage.ray.get_actor("other", namespace="unrelated")
    ray.get_actor.assert_called_with("other", namespace="unrelated")
    assert ray.get is interface.ray.get
    assert ray is not interface.ray
    monkeypatch.setenv("VERL_TRANSFER_QUEUE_NAMESPACE", "evaluation-40")
    with pytest.raises(RuntimeError, match="initialized TransferQueue"):
        module.configure_transfer_queue_namespace()


def test_disabled_does_not_import_dependency(monkeypatch):
    monkeypatch.delenv("VERL_TRANSFER_QUEUE_NAMESPACE", raising=False)
    imported = Mock(side_effect=AssertionError("must not import"))
    monkeypatch.setattr(module.importlib, "import_module", imported)
    module.configure_transfer_queue_namespace()
    imported.assert_not_called()


def test_refuses_initialized_queue(monkeypatch):
    interface = SimpleNamespace(_TQ_CONTROLLER=object())
    monkeypatch.setattr(module.importlib, "import_module", lambda _: interface)
    monkeypatch.setenv("VERL_TRANSFER_QUEUE_NAMESPACE", "evaluation-20")
    with pytest.raises(RuntimeError, match="initialized TransferQueue"):
        module.configure_transfer_queue_namespace()
