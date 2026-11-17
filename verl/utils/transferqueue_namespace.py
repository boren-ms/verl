# Copyright 2026 Individual Contributor: Boren
# SPDX-License-Identifier: Apache-2.0
"""Optional per-job isolation for TransferQueue releases with a fixed namespace."""

import importlib
import os


class _NamespacedActor:
    def __init__(self, actor, namespace):
        self._actor = actor
        self._namespace = namespace

    def options(self, **kwargs):
        if kwargs.get("namespace") in (None, "transfer_queue"):
            kwargs["namespace"] = self._namespace
        return self._actor.options(**kwargs)

    def __getattr__(self, name):
        return getattr(self._actor, name)


class _NamespacedRay:
    def __init__(self, ray, namespace):
        self._ray = ray
        self._namespace = namespace

    def get_actor(self, name, **kwargs):
        if kwargs.get("namespace") == "transfer_queue":
            kwargs["namespace"] = self._namespace
        return self._ray.get_actor(name, **kwargs)

    def __getattr__(self, name):
        return getattr(self._ray, name)


def configure_transfer_queue_namespace():
    """Scope only TQ's named actors; leave Ray and installed packages unchanged.

    Call before tq.init() in drivers and workers. With the variable unset, the
    dependency's original behavior is retained. The shim is process-local and
    idempotent, and refuses to switch an already initialized queue.
    """
    namespace = os.environ.get("VERL_TRANSFER_QUEUE_NAMESPACE")
    if not namespace:
        return
    interface = importlib.import_module("transfer_queue.interface")
    previous = getattr(interface, "_verl_namespace", None)
    if previous == namespace:
        return
    if previous is not None or any(
        getattr(interface, name, None) is not None for name in ("_TQ_CONTROLLER", "_TQ_CLIENT", "_TQ_STORAGE")
    ):
        raise RuntimeError("Cannot change the namespace of an initialized TransferQueue")
    interface.ray = _NamespacedRay(interface.ray, namespace)
    interface.TransferQueueController = _NamespacedActor(interface.TransferQueueController, namespace)
    storage = importlib.import_module("transfer_queue.storage.clients.ray_storage_client")
    storage.ray = _NamespacedRay(storage.ray, namespace)
    storage.RayObjectRefStorage = _NamespacedActor(storage.RayObjectRefStorage, namespace)
    bootstrap = importlib.import_module("transfer_queue.storage.bootstrap.simple_storage_bootstrap")
    bootstrap.SimpleStorageUnit = _NamespacedActor(bootstrap.SimpleStorageUnit, namespace)
    interface._verl_namespace = namespace
