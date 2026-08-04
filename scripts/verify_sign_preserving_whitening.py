#!/usr/bin/env python3

import torch


def rms_normalize(vector: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return vector / torch.sqrt(vector.square().mean() + eps)


def l2_normalize(vector: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return vector / (torch.linalg.vector_norm(vector) + eps)


def verify(name: str, original: torch.Tensor, normalized: torch.Tensor, target: str) -> None:
    torch.testing.assert_close(torch.sign(normalized), torch.sign(original), rtol=0, atol=0)

    if target == "rms":
        scale = torch.sqrt(normalized.square().mean())
    else:
        scale = torch.linalg.vector_norm(normalized)

    torch.testing.assert_close(scale, torch.ones_like(scale), rtol=1e-6, atol=1e-6)
    print(f"{name}: PASS")
    print(f"  input:  {original.tolist()}")
    print(f"  output: {normalized.tolist()}")
    print(f"  signs:  {torch.sign(normalized).tolist()}")
    print(f"  {target}: {scale.item():.8f}")


def main() -> None:
    vector = torch.tensor([-4.0, -1.5, 0.0, 2.0, 7.0], dtype=torch.float64)
    verify("RMS normalization", vector, rms_normalize(vector), "rms")
    verify("L2 normalization", vector, l2_normalize(vector), "l2")


if __name__ == "__main__":
    main()