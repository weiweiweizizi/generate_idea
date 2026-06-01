#!/usr/bin/env python
from __future__ import annotations

import fire


def train(**kwargs):
    from disentangleNet.training import train

    return train(**kwargs)


if __name__ == "__main__":
    fire.Fire(train)
