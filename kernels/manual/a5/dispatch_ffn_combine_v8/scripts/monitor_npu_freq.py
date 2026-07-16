#!/usr/bin/env python3
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Continuously monitor Ascend AI Core frequency through libdcmi.so."""

import argparse
import ctypes
import csv
import datetime
import os
import sys
import time
from ctypes import POINTER, Structure, byref, c_int, c_uint
from pathlib import Path


FREQ_AICORE_CURRENT = 7
FREQ_AICORE_MAX = 9
DEFAULT_LIBRARY_PATHS = (
    "libdcmi.so",
    "/usr/local/Ascend/driver/lib64/driver/libdcmi.so",
    "/usr/local/Ascend/driver/lib64/libdcmi.so",
)


class AICoreInfo(Structure):
    _fields_ = [("freq", c_uint), ("cur_freq", c_uint)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor Ascend AI Core frequency without npu-smi by calling libdcmi.so directly."
    )
    parser.add_argument("--card", type=int, help="Only monitor this DCMI card ID; default: all cards")
    parser.add_argument("--device", type=int, help="Only monitor this device ID; requires --card")
    parser.add_argument("--interval", type=float, default=0.1, help="Sampling interval in seconds; default: 0.1")
    parser.add_argument("--count", type=int, default=0, help="Number of samples; 0 means run until Ctrl-C")
    parser.add_argument("--csv", action="store_true", help="Emit CSV instead of human-readable text")
    parser.add_argument("--library", type=Path, help="Explicit path to libdcmi.so")
    args = parser.parse_args()
    if args.device is not None and args.card is None:
        parser.error("--device requires --card")
    if args.interval <= 0:
        parser.error("--interval must be positive")
    if args.count < 0:
        parser.error("--count must be non-negative")
    return args


def load_dcmi(explicit_path: Path | None) -> tuple[ctypes.CDLL, str]:
    candidates = []
    env_path = os.environ.get("DCMI_LIBRARY_PATH")
    if explicit_path is not None:
        candidates.append(str(explicit_path))
    if env_path:
        candidates.append(env_path)
    candidates.extend(DEFAULT_LIBRARY_PATHS)

    errors = []
    for candidate in candidates:
        try:
            return ctypes.CDLL(candidate), candidate
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")

    detail = "\n  ".join(errors)
    raise RuntimeError(
        f"libdcmi.so was not found. Mount/install the Ascend driver library or pass --library.\nTried:\n  {detail}"
    )


def configure_api(dcmi: ctypes.CDLL) -> None:
    dcmi.dcmi_init.argtypes = []
    dcmi.dcmi_init.restype = c_int

    card_list_fn = getattr(dcmi, "dcmi_get_card_list", None)
    if card_list_fn is None:
        card_list_fn = getattr(dcmi, "dcmi_get_card_num_list", None)
    if card_list_fn is None:
        raise RuntimeError("libdcmi.so has neither dcmi_get_card_list nor dcmi_get_card_num_list")
    card_list_fn.argtypes = [POINTER(c_int), POINTER(c_int), c_int]
    card_list_fn.restype = c_int
    dcmi.card_list_fn = card_list_fn

    dcmi.dcmi_get_device_num_in_card.argtypes = [c_int, POINTER(c_int)]
    dcmi.dcmi_get_device_num_in_card.restype = c_int

    frequency_fn = getattr(dcmi, "dcmi_get_device_frequency", None)
    if frequency_fn is not None:
        frequency_fn.argtypes = [c_int, c_int, c_int, POINTER(c_uint)]
        frequency_fn.restype = c_int
    dcmi.frequency_fn = frequency_fn

    aicore_info_fn = getattr(dcmi, "dcmi_get_device_aicore_info", None)
    if aicore_info_fn is None:
        aicore_info_fn = getattr(dcmi, "dcmi_get_aicore_info", None)
    if aicore_info_fn is not None:
        aicore_info_fn.argtypes = [c_int, c_int, POINTER(AICoreInfo)]
        aicore_info_fn.restype = c_int
    dcmi.aicore_info_fn = aicore_info_fn

    for name, value_type in (
        ("dcmi_get_device_power_info", c_int),
        ("dcmi_get_device_temperature", c_int),
        ("dcmi_get_device_voltage", c_uint),
    ):
        function = getattr(dcmi, name, None)
        if function is not None:
            function.argtypes = [c_int, c_int, POINTER(value_type)]
            function.restype = c_int


def check_ret(operation: str, ret: int) -> None:
    if ret != 0:
        raise RuntimeError(f"{operation} failed with DCMI error {ret}")


def enumerate_devices(dcmi: ctypes.CDLL, card_filter: int | None, device_filter: int | None) -> list[tuple[int, int]]:
    card_num = c_int()
    card_ids = (c_int * 64)()
    check_ret("dcmi_get_card_list", dcmi.card_list_fn(byref(card_num), card_ids, len(card_ids)))

    devices = []
    for card_id in list(card_ids)[: card_num.value]:
        if card_filter is not None and card_id != card_filter:
            continue
        device_num = c_int()
        check_ret(
            f"dcmi_get_device_num_in_card(card={card_id})", dcmi.dcmi_get_device_num_in_card(card_id, byref(device_num))
        )
        for device_id in range(device_num.value):
            if device_filter is None or device_id == device_filter:
                devices.append((card_id, device_id))

    if not devices:
        raise RuntimeError(f"no matching DCMI device found for card={card_filter}, device={device_filter}")
    return devices


def query_scalar(function, value_type, card_id: int, device_id: int) -> tuple[int | None, int]:
    if function is None:
        return None, -1
    value = value_type()
    ret = function(card_id, device_id, byref(value))
    return (value.value if ret == 0 else None), ret


def query_frequency(dcmi: ctypes.CDLL, card_id: int, device_id: int) -> tuple[int | None, int | None, int]:
    if dcmi.frequency_fn is not None:
        current = c_uint()
        maximum = c_uint()
        current_ret = dcmi.frequency_fn(card_id, device_id, FREQ_AICORE_CURRENT, byref(current))
        maximum_ret = dcmi.frequency_fn(card_id, device_id, FREQ_AICORE_MAX, byref(maximum))
        if current_ret == 0 and maximum_ret == 0:
            return current.value, maximum.value, 0

    if dcmi.aicore_info_fn is not None:
        info = AICoreInfo()
        ret = dcmi.aicore_info_fn(card_id, device_id, byref(info))
        if ret == 0:
            return info.cur_freq, info.freq, 0
        return None, None, ret

    return None, None, -1


def query_device(dcmi: ctypes.CDLL, card_id: int, device_id: int) -> dict[str, int | float | str | None]:
    current_mhz, max_mhz, frequency_ret = query_frequency(dcmi, card_id, device_id)
    power_raw, power_ret = query_scalar(getattr(dcmi, "dcmi_get_device_power_info", None), c_int, card_id, device_id)
    temperature_c, temperature_ret = query_scalar(
        getattr(dcmi, "dcmi_get_device_temperature", None), c_int, card_id, device_id
    )
    voltage_raw, voltage_ret = query_scalar(getattr(dcmi, "dcmi_get_device_voltage", None), c_uint, card_id, device_id)
    return {
        "time": datetime.datetime.now().isoformat(timespec="milliseconds"),
        "card": card_id,
        "device": device_id,
        "aicore_current_mhz": current_mhz,
        "aicore_max_mhz": max_mhz,
        "power_w": power_raw / 10.0 if power_raw is not None else None,
        "temperature_c": temperature_c,
        "voltage_v": voltage_raw / 100.0 if voltage_raw is not None else None,
        "frequency_ret": frequency_ret,
        "power_ret": power_ret,
        "temperature_ret": temperature_ret,
        "voltage_ret": voltage_ret,
    }


def format_value(value, suffix: str = "") -> str:
    return "NA" if value is None else f"{value}{suffix}"


def print_text(sample: dict[str, int | float | str | None]) -> None:
    print(
        f"{sample['time']} card={sample['card']} dev={sample['device']} "
        f"aicore={format_value(sample['aicore_current_mhz'], 'MHz')} "
        f"max={format_value(sample['aicore_max_mhz'], 'MHz')} "
        f"power={format_value(sample['power_w'], 'W')} "
        f"temp={format_value(sample['temperature_c'], 'C')} "
        f"volt={format_value(sample['voltage_v'], 'V')} "
        f"ret(freq/power/temp/volt)={sample['frequency_ret']}/{sample['power_ret']}/"
        f"{sample['temperature_ret']}/{sample['voltage_ret']}",
        flush=True,
    )


def monitor(dcmi: ctypes.CDLL, devices: list[tuple[int, int]], interval: float, count: int, csv_output: bool) -> None:
    writer = None
    if csv_output:
        fieldnames = [
            "time",
            "card",
            "device",
            "aicore_current_mhz",
            "aicore_max_mhz",
            "power_w",
            "temperature_c",
            "voltage_v",
            "frequency_ret",
            "power_ret",
            "temperature_ret",
            "voltage_ret",
        ]
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()

    iteration = 0
    while count == 0 or iteration < count:
        start = time.monotonic()
        for card_id, device_id in devices:
            sample = query_device(dcmi, card_id, device_id)
            if writer is not None:
                writer.writerow(sample)
                sys.stdout.flush()
            else:
                print_text(sample)
        iteration += 1
        if count != 0 and iteration >= count:
            break
        remaining = interval - (time.monotonic() - start)
        if remaining > 0:
            time.sleep(remaining)


def main() -> int:
    args = parse_args()
    try:
        dcmi, library_path = load_dcmi(args.library)
        configure_api(dcmi)
        check_ret("dcmi_init", dcmi.dcmi_init())
        devices = enumerate_devices(dcmi, args.card, args.device)
        if not args.csv:
            print(f"libdcmi={library_path} devices={devices} interval={args.interval}s", flush=True)
        monitor(dcmi, devices, args.interval, args.count, args.csv)
    except KeyboardInterrupt:
        return 0
    except (AttributeError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
