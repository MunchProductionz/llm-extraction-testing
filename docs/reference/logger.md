# `logger`

This module defines `RunLogger`, the package's human-readable text log writer.

## `RunLogger`

Constructor:

- `RunLogger(log_directory_path: str)`

Behavior:

- stores the target directory path
- ensures the directory exists immediately

Main method:

- `write_log(run_context, result_bundle, run_config, note_message=None) -> str`

Returns:

- the path to the written log file as a string

Side effects:

- creates the log directory if needed
- writes a text file to disk

Error conditions:

- the module does not perform much explicit validation; file-system errors would surface from the underlying write path

## Generated API details

::: extraction_testing.logger
    options:
      members:
        - RunLogger
      show_root_heading: false
