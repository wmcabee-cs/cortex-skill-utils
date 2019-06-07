#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from cortex_skill_utils.skeleton import fib

__author__ = "William McAbee"
__copyright__ = "William McAbee"
__license__ = "mit"


def test_fib():
    assert fib(1) == 1
    assert fib(2) == 1
    assert fib(7) == 13
    with pytest.raises(AssertionError):
        fib(-10)
