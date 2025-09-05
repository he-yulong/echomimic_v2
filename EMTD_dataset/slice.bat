@echo off
REM --- slice.bat: run PySceneDetect on all videos ---

REM 1) Where the videos are (input) and where to put the clips (output)
set "ROOT=%~dp0"
set "TARGET=ori_video_dir"
set "OUTDIR=%ROOT%%TARGET%_segs"

REM 2) Ensure output folder exists
if not exist "%OUTDIR%" mkdir "%OUTDIR%" 2>nul

REM 3) Call scenedetect
for %%F in ("%ROOT%%TARGET%\*.mp4") do (
  echo Processing %%~nxF ...
  scenedetect --input "%%~fF" detect-content split-video -o "%OUTDIR%"
)

echo Done. Clips saved to: %OUTDIR%