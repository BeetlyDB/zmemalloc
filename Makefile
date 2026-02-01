util_fast:
	@zig test src/util.zig -OReleaseFast



test_fast:
	@zig build test -Doptimize=ReleaseFast
