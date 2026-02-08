util_fast:
	@zig test src/util.zig -OReleaseFast



test_fast:
	@zig build test -Doptimize=ReleaseFast



perf:
	@perf record -F 999 -g -call-graph dwarf ./zig-out/bin/zmemalloc



bench:
	@zig build bench -Doptimize=ReleaseFast 2>&1

