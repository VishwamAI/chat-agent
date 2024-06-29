# Pull Request: Enable Debug Mode for Sympify Operation Profiling

## Description

This pull request enables the debug mode for the sympify operation profiling in the `architecture.py` file. The debug mode is set to `True` in the `_batch_sympify` function to allow for detailed profiling of the sympify operation, including memory usage and execution time.

## Changes Made

- Modified the `_batch_sympify` function in `src/model/architecture.py` to set the `debug` flag to `True`.

## Rationale

Enabling the debug mode will help us collect profiling data to identify bottlenecks in the sympify operation. This data will be used to optimize the sympify operation for better performance and efficiency.

## Next Steps

1. Run the code to collect profiling data.
2. Analyze the profiling data to identify bottlenecks.
3. Make further optimizations based on the profiling data.

## Related Issues

N/A

## Checklist

- [x] Enable debug mode for sympify operation profiling.
- [x] Commit changes to a new branch.
- [x] Push the new branch to the remote repository.
- [x] Create a pull request for the changes.

## Additional Notes

Please review the changes and provide feedback. Once the profiling data is collected and analyzed, further optimizations will be made to improve the performance of the sympify operation.

[This Devin run](https://preview.devin.ai/devin/741f66af4da743ce8cfb4b2cd1467f58) was requested by kasinadhsarma.
