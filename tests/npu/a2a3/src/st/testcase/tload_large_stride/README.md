# TLOAD stride ST

The host harness creates deterministic inputs and checks loaded values through public `TLOAD` calls.
No external data generation is required.

- `TLoadLargeStrideTest`: 23 cases for ND/DN/NZ, static/dynamic strides, two batches,
  destination gaps, 16-bit L1 gap boundaries, 32-bit UB gap boundaries, 64-bit outer
  strides, and partial-row padding. The large cases need slightly more than 4 GiB
  of device memory, released after each case.
- `TLoadConversionStrideTest`: 24 b32 ND2NZ/DN2ZN cases. `float`, `int32_t` and
  `uint32_t` use element strides 32767, 32768 and 65535. Both sides of the b16
  conversion boundary and static/dynamic stride paths are checked. Each case
  allocates less than 257 KiB of input and checks both destination C0 blocks.
- `TLoadVectorLengthTest`: 48 A2/A3 and KirinX90 cases for single-row ND and
  single-column DN Mat loads. b8/b16/b32 lengths cross 65536 elements; b64 lengths
  cross the b16/b32 conversion boundaries at 16384 and 32768 elements. Static and
  dynamic valid lengths are covered. Tests check every loaded element, zero
  padding up to the next 32-byte boundary, and an adjacent block left unchanged.
  Inputs after the valid range contain nonzero values to detect copying them into
  destination padding. These value checks do not prove the absence of extra hardware reads.
  Eight cases offset the GM source by one element to check unaligned addresses.
- `TLoadWideStrideTest`: A5/A6 cases for byte strides below, at and above 2^40,
  plus Shape1/Shape2 hardware loops. A5 also checks the NC1HWC0 convolution loop.
  Virtual memory maps only accessed pages; physical allocation is a few pages,
  but the device must support reserving a virtual address span greater than 1 TiB.
  Failure to reserve/map memory fails the test before kernel execution.

Registration depends on the backend selected by CMake:

| Backend | Large stride | Conversion stride | Vector length | Wide stride | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| A2/A3 | 23 | 24 | 48 | 0 | 95 |
| A5 | 23 | 24 | 0 | 27 | 74 |
| A6 | 11 | 0 | 0 | 13 | 24 |
| KirinX90 | 23 | 24 | 48 | 0 | 95 |
| Kirin9030 | 23 | 24 | 0 | 0 | 47 |
| KirinDev0000 | 11 | 0 | 0 | 0 | 11 |

These are registered case counts, not runtime pass counts. A2/A3, A5, KirinX90 and
Kirin9030 share the base L1/UB and conversion suites. A6 and KirinDev0000 use only the
UB subset because their L1 readback paths differ. A5 has 13 Mat and 13 Vec wide-stride
cases plus one NC1HWC0 case; A6 has the 13 Vec cases. The single-row/single-column
length suite is registered only for the A2/A3 and KirinX90 memory backends.

Small and below-boundary cases remain as controls for the fast paths. Above-boundary
cases check fallbacks on backends whose instruction fields exceed their range. The same
case can stay on a native path on another backend: for example, a source stride just
above 4 GiB fits the A5/A6 40-bit field. Both groups are needed to catch regressions. The 40-bit cases
are registered only for A5/A6 and require working device VMM support for the sparse span.
A VMM allocation failure does not establish whether the kernel fallback is correct.

The table covers only `tload_large_stride`. Additional regression cases in existing suites are:

- A2/A3 [`tload_gm2mat`](../../../../../a2a3/src/st/testcase/tload_gm2mat): three NZ cases with source gaps of
  65408, 65536 and 65408 blocks (32 bytes per block), plus four gap-overflow cases for
  NC1HWC0, FRACTAL_Z (two representations) and NDC1HWC0.
- Kirin9030 [`tload_mix`](../../../../../kirin9030/src/st/testcase/tload_mix), also
  shared by KirinX90: three NZ cases around the block-gap boundary, including a destination
  row gap, and three ND cases for unaligned source strides and the byte-gap boundary.

These existing suites use `gen_data.py`; run them separately with `-t tload_gm2mat` or
`-t tload_mix` on the corresponding backend. Their cases are not included in the table above.

The [TLOAD interface](../../../../../../../docs/isa/TLOAD.md) documents the distinct gap,
stride and format-conversion constraints.

Use the normal ST runner with `-t tload_large_stride`. GTest filters can select the
suites independently. Kernel compilation does not establish runtime correctness.
