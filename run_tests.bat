@echo off
REM Set PYTHONPATH so "src" can be imported correctly
SET PYTHONPATH=%CD%

REM Create the test_reports directory if it doesn't exist
IF NOT EXIST test_reports (
    mkdir test_reports
)

REM Run pytest and write results to a log file
pytest tests/ --maxfail=1 --disable-warnings -q > test_reports/test_log.txt

REM Optional: Display success message
echo Tests completed. See test_reports\test_log.txt for details.
