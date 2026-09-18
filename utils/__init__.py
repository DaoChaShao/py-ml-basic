#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/18 22:17
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   __init__.py
# @Desc     :   

"""
****************************************************************
Utility Module - Comprehensive Toolkit
----------------------------------------------------------------
This module provides a comprehensive suite of utility functions
and classes designed for general data processing tasks.
****************************************************************
"""

from importlib.metadata import metadata, PackageNotFoundError

try:
    _meta = metadata("py-ml-basic")
    __version__ = _meta.get("Version", "0.0.0")
    __author__ = _meta.get("Author", "Shawn Yu")
except PackageNotFoundError:
    __author__ = "Shawn Yu"
    __version__ = "0.1.0"

from .decorator import beautifier, clock, countdown, timer
from .helper import Beautifier, RandomSeed, Timer
from .highlighter import (
    black, red, green, yellow, blue, purple, cyan, white,
    bold, underline, invert, strikethrough,
    stars, lines, sharps
)

__all__ = [
    "beautifier",
    "timer", "clock",
    "countdown",

    "Beautifier",
    "Timer",
    "RandomSeed",

    "black", "red", "green", "yellow", "blue", "purple", "cyan", "white",
    "bold", "underline", "invert", "strikethrough",
    "stars", "lines", "sharps",
]
