# start/start-here.ps1 [PACK [MODEL_FOLDER]] - NanoSeedLM's start on a Windows PC (START-HERE.bat runs it): check the PC,
# pick a pack, download and check it, chat once, write run-PACK.bat and start the server. Run it again to resume.
param([string]$Pack, [string]$ModelDir)   # given: no questions
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false   # PowerShell 7: the script reads curl's and nslm-chat's exit codes
$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$src = $false   # a source checkout keeps this script in start\: the engine folder is the checkout, and the script builds it
$up = Split-Path -Parent $here
if ((Test-Path "$up\start\start-here.ps1") -and (Test-Path "$up\win\build.bat") -and (Test-Path "$up\CMakeLists.txt")) { $here = $up; $src = $true }
Set-Location $here
$hfile = $null   # the token for curl, as a header file
function Fail($m) {
    if ($script:hfile) { Remove-Item -Force -ErrorAction SilentlyContinue $script:hfile }
    Write-Host "`n$m" -ForegroundColor Red; exit 1
}

# name = repository, revision, least RAM GB, least VRAM GB, download GB (the model cards' "For" column)
$packs = [ordered]@{
    'TCQ4'    = @('txgsync/Qwen3.8-Flash-Next-NanoSeedLM-TCQ4', '37f93190f0220c749ff5bd81b2153b02bf3c4b49', 96, 24, 128)
    'TCQ3-D4' = @('txgsync/Qwen3.8-Flash-Next-NanoSeedLM-TCQ3-D4', 'b3bc4f82a7b6d0abfaec07f8619fd5256be97aa2', 96, 16, 117)
    'TCQ3'    = @('txgsync/Qwen3.8-Flash-Next-NanoSeedLM-TCQ3', '603685c39c0328133872f0c6f9a657bf81def07e', 64, 12, 112)
}
if ($env:NSLM_PACKS) {   # another table, as start-here.sh reads it: lines "name repository revision RAM VRAM MacGB downloadGB"
    $packs = [ordered]@{}
    foreach ($l in ($env:NSLM_PACKS -split "`n")) {
        $w = -split $l
        if ($w.Count -ge 7) { $packs[$w[0]] = @($w[1], $w[2], [int]$w[3], [int]$w[4], [int]$w[6]) }
    }
}

Write-Host "NanoSeedLM: Qwen3.8-Flash-Next on this PC`n"
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { Fail "No NVIDIA driver found. Install driver 570 or newer: https://www.nvidia.com/drivers" }
$g = ((& nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits) | Select-Object -First 1).Split(',') | ForEach-Object { $_.Trim() }
$vram = [math]::Round([double]$g[1] / 1024)
$ram = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
Write-Host "GPU: $($g[0]), $vram GB, driver $($g[2]). RAM: $ram GB."
if ([int]$g[2].Split('.')[0] -lt 570) { Fail "The driver is older than 570 (CUDA 12.8). Update it: https://www.nvidia.com/drivers" }
if ($vram -lt 12) { Write-Host "Warning: less than 12 GB of VRAM. The engine may not fit." -ForegroundColor Yellow }

# the engine (nslm-chat, nslm-serve), as in start-here.sh. The signed release of this version, when it is built from
# this engine source: a release folder, or a checkout at the release's engine source. A checkout with other engine
# source builds it (win\build.bat), and downloads the release only when it has no compiler or the build fails.
$gh = 'https://github.com/mbarnson/NanoSeedLM/releases'
$tag = $null
if (Test-Path VERSION) { $tag = 'v' + (Get-Content -Raw VERSION).Trim() }
elseif ((Test-Path README.txt) -and ((Get-Content README.txt -TotalCount 1) -match '^NanoSeedLM engine (v[0-9][0-9.]*):')) { $tag = $Matches[1] }
function Have { (Test-Path build\bin\nslm-serve.exe) -and (Test-Path build\bin\nslm-chat.exe) }
function CanCC {   # Visual Studio's C++ tools and CUDA. win\build.bat also needs CMake and Ninja
    $vsw = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vsw)) { return $false }
    $vs = & $vsw -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    [bool]$vs -and ([bool](Get-Command nvcc -ErrorAction SilentlyContinue) -or [bool]$env:CUDA_PATH)
}
function Build {   # again when the source changes. The first build is for this GPU only: a shorter build
    Write-Host "Building the engine from this checkout (win\build.bat: Visual Studio and CUDA, some minutes the first time)..."
    if (-not (Test-Path build\build.ninja) -and -not $env:NSLM_CMAKE_ARGS) { $env:NSLM_CMAKE_ARGS = '-DCMAKE_CUDA_ARCHITECTURES=native' }
    & cmd /c "win\build.bat" | Out-Host
    ($LASTEXITCODE -eq 0) -and (Have)
}
function Fetch {   # the release zip of $tag, checked against the release's SHA256SUMS. Only its build\bin is copied
    if (-not $tag) { Write-Host "This folder does not name its version (VERSION or README.txt)."; return $false }
    $a = "nslm-$tag-windows-x64-cuda12.8"
    $t = Join-Path ([IO.Path]::GetTempPath()) ([IO.Path]::GetRandomFileName())
    New-Item -ItemType Directory $t | Out-Null
    Write-Host "Downloading the engine $tag from GitHub ($a.zip)..."
    & curl.exe -fsSL --retry 5 -o "$t\$a.zip" "$gh/download/$tag/$a.zip"
    if (-not $LASTEXITCODE) { & curl.exe -fsSL --retry 5 -o "$t\SHA256SUMS" "$gh/download/$tag/SHA256SUMS" }
    if ($LASTEXITCODE) { Remove-Item -Recurse -Force $t; Write-Host "Cannot download the engine $tag from $gh."; return $false }
    $want = $null
    foreach ($l in Get-Content "$t\SHA256SUMS") { $w = $l -split '\s+\*?'; if ($w.Count -ge 2 -and $w[1] -eq "$a.zip") { $want = $w[0] } }
    if (-not $want -or (Get-FileHash -Algorithm SHA256 "$t\$a.zip").Hash.ToLower() -ne $want) {
        Remove-Item -Recurse -Force $t
        Fail "The engine archive $a.zip failed its SHA-256 check. Run START-HERE.bat again."
    }
    Expand-Archive -Path "$t\$a.zip" -DestinationPath "$t\x"
    New-Item -ItemType Directory -Force build\bin | Out-Null
    Copy-Item -Force "$t\x\$a\build\bin\*" build\bin\
    Remove-Item -Recurse -Force $t
    Have
}
$same = $false   # a checkout whose engine source is the source of the release $tag
if ($src -and $tag -and (Test-Path .git) -and (Get-Command git -ErrorAction SilentlyContinue)) {
    $null = & git rev-parse -q --verify "refs/tags/$tag^{commit}"
    if (-not $LASTEXITCODE) { $null = & git diff --quiet $tag -- engine harness nslm win Makefile CMakeLists.txt; $same = -not $LASTEXITCODE }
}
$nobuild = "Build the engine (docs\guides\build.md), or download it from $gh."
if ($src -and -not $same -and (CanCC)) {   # other engine source than the release
    if (-not (Build)) {
        Write-Host "The build failed (see above). The release $tag comes from an earlier source. Downloading it instead."
        if (-not (Fetch)) { Fail $nobuild }
    }
} elseif (-not (Have)) {
    if ($src -and -not $same) { Write-Host "No compiler for a build. The release $tag comes from an earlier source." }
    if (-not (Fetch)) { if (-not ($src -and (CanCC) -and (Build))) { Fail $nobuild } }
}
if (-not (Have)) { Fail "build\bin\nslm-serve.exe is missing. $nobuild" }

function Fits($have, $size) { $have -ge 0.9 * $size }   # a 64 GB PC may report 62 GB: 90% of a size counts as that size
$rec = $null
foreach ($n in $packs.Keys) { $p = $packs[$n]; if (-not $rec -and (Fits $ram $p[2]) -and (Fits $vram $p[3])) { $rec = $n } }
if (-not $rec) { if (-not (Fits $ram 64)) { Fail "The smallest pack (TCQ3) needs 64 GB of RAM. This PC has $ram GB." }; $rec = 'TCQ3' }
Write-Host "`nPacks (KL against BF16: lower is closer to the original model):"
if ($env:NSLM_PACKS) { foreach ($n in $packs.Keys) { Write-Host "  $n  ($($packs[$n][4]) GB download)" } }
else {
    Write-Host "  TCQ4     0.0415  for a GPU of 24 GB or more with 96 to 128 GB of RAM ($($packs['TCQ4'][4]) GB download)"
    Write-Host "  TCQ3-D4  0.0638  for a 16 GB GPU with 96 to 128 GB of RAM ($($packs['TCQ3-D4'][4]) GB download)"
    Write-Host "  TCQ3     0.0778  for a 12 or 16 GB GPU with 64 GB of RAM ($($packs['TCQ3'][4]) GB download)"
}
Write-Host "Each pack carries its PLE table at 8 bits. It stays on the disk, and the page cache holds the rows in use."
$pick = $Pack
if (-not $pick) { $pick = Read-Host "Pack [$rec]" }
if (-not $pick) { $pick = $rec }
if (-not $packs.Contains($pick)) { Fail "Unknown pack $pick. Use TCQ4, TCQ3-D4 or TCQ3." }
$root = $ModelDir
if (-not $root -and -not $Pack) { $root = Read-Host "Model folder [$here\models]" }
if (-not $root) { $root = "$here\models" }
if ($root -eq '~' -or $root.StartsWith('~\') -or $root.StartsWith('~/')) { $root = $HOME + $root.Substring(1) }   # curl.exe keeps ~
$repo, $rev = $packs[$pick][0], $packs[$pick][1]
$dir = Join-Path $root ($repo.Split('/')[1])
New-Item -ItemType Directory -Force $dir | Out-Null

# the files at their pinned revisions, each checked by size and SHA-256 (verified ones are listed in .nslm-verified)
# The token: HF_TOKEN, or the one that "hf auth login" saved (a signed-in download has higher rate limits). curl reads
# it from a file, so it is not in the process list.
$tok = $env:HF_TOKEN
$tf = $(if ($env:HF_TOKEN_PATH) { $env:HF_TOKEN_PATH } elseif ($env:HF_HOME) { Join-Path $env:HF_HOME 'token' } else { Join-Path $HOME '.cache\huggingface\token' })
if (-not $tok -and (Test-Path $tf)) { $tok = (Get-Content -Raw $tf).Trim() }
$hdr = @{}
$hfile = [IO.Path]::GetTempFileName()
if ($tok) { $hdr['Authorization'] = "Bearer $tok"; [IO.File]::WriteAllText($hfile, "Authorization: Bearer $tok") }
# .nslm-verified lists "path size id" for each file checked at this revision (id: a large file's SHA-256, git:BLOB for a
# small one). An earlier revision's line or v0.1.0's bare path matches nothing: then a large file of the right size and
# SHA-256 stays, and any other file downloads again.
function Files($r, $v) {   # the files of repository r at revision v
    $u = "https://huggingface.co/api/models/$r/tree/$($v)?recursive=true"
    $t = Invoke-RestMethod -Headers $hdr $u   # PowerShell 5.1: one array object
    @($t | Where-Object { $_.type -eq 'file' -and -not $_.path.StartsWith('.') } | ForEach-Object {
        $sha = $(if ($_.lfs) { $_.lfs.oid })
        [pscustomobject]@{ path = $_.path; size = $_.size; sha = $sha; id = $(if ($sha) { $sha } else { "git:$($_.oid)" }) } })
}
function Line($f) { "$($f.path) $($f.size) $($f.id)" }
$files = @(Files $repo $rev)
$okf = Join-Path $dir '.nslm-verified'
$old = @(if (Test-Path $okf) { Get-Content $okf })
$ok = @{}
foreach ($l in $old) { $ok[$l] = 1 }
$new = [System.Collections.Generic.List[string]]::new()
$check, $need, $hashed = 0, 0, $false
foreach ($f in $files) {
    if ($f.path.StartsWith('/') -or $f.path.Contains('..')) { Fail "The pack lists an unsafe path: $($f.path)" }
    $dst = Join-Path $dir $f.path
    if ($ok.ContainsKey((Line $f)) -and (Test-Path $dst) -and (Get-Item $dst).Length -eq $f.size) { $new.Add((Line $f)); continue }   # checked before
    if ($f.sha -and (Test-Path $dst) -and (Get-Item $dst).Length -eq $f.size) {   # from an earlier download
        if (-not $hashed) { Write-Host "Checking the files of an earlier download (SHA-256)..."; $hashed = $true }
        if ((Get-FileHash -Algorithm SHA256 $dst).Hash.ToLower() -eq $f.sha) { $new.Add((Line $f)); continue }
    }
    if (Test-Path $dst) { Remove-Item $dst }   # an older or broken version: a download starts over, never through a hard link
    $check += 1
    $have = 0
    if (Test-Path "$dst.part") { $have = (Get-Item "$dst.part").Length }
    if ($have -gt $f.size) { Remove-Item "$dst.part"; $have = 0 }
    $need += $f.size - $have
}
Set-Content $okf $new
$paths = @{}
foreach ($f in $files) { $paths[$f.path] = 1 }
$gone = 0   # the files of an earlier download that this revision does not have
foreach ($l in $old) {
    $p = ($l -split ' ')[0]
    if (-not $p -or $p.StartsWith('/') -or $p.Contains('..') -or $paths.ContainsKey($p)) { continue }
    $q = Join-Path $dir $p
    if (Test-Path $q) { $gone += (Get-Item $q).Length; Remove-Item $q }
}
if ($gone) { Write-Host ("Removed {0:N1} GB of earlier files that this revision of the pack does not have." -f ($gone / 1e9)) }
$free = (Get-Item $dir).PSDrive.Free
if ($need -gt $free) { Fail ("The download needs {0:N0} GB and the disk has {1:N0} GB free." -f ($need / 1e9), ($free / 1e9)) }
Write-Host ("`n{0} files to download, {1:N1} GB into {2}" -f $check, ($need / 1e9), $root)
if ($need -gt 1GB -and -not $Pack -and (Read-Host "Start? [Y/n]") -match '^[nN]') { Write-Host "Stopped. Run START-HERE.bat again."; exit 0 }
if ($need -gt 1GB -and $Pack) { Write-Host "The download starts in 5 s (Ctrl+C stops it)."; Start-Sleep 5 }   # a folder typo
# The download, as in start-here.sh: NSLM_DL_JOBS files at once (8). hf, when it is installed (NSLM_NO_HF=1: not), then
# curl for each file that remains. Each file is checked by its size and SHA-256 before it joins .nslm-verified.
$jobs = $(if ($env:NSLM_DL_JOBS) { [int]$env:NSLM_DL_JOBS } else { 8 })
function Good($p, $f) {   # file p has f's size, and a large file has its SHA-256
    (Test-Path $p) -and ((Get-Item $p).Length -eq $f.size) -and (-not $f.sha -or (Get-FileHash -Algorithm SHA256 $p).Hash.ToLower() -eq $f.sha)
}
function Todo {   # the files that are not in .nslm-verified
    $v = @{}
    if (Test-Path $okf) { foreach ($l in Get-Content $okf) { $v[$l] = 1 } }
    @($files | Where-Object { -not $v.ContainsKey((Line $_)) })
}
$todo = @(Todo)
$hf = (-not $env:NSLM_NO_HF) -and [bool](Get-Command hf -ErrorAction SilentlyContinue)
if ($todo.Count -and -not $hf -and -not $env:NSLM_NO_HF) {
    Write-Host "Tip: hf can download faster. It uses Hugging Face's Xet transfers and your login. To install it:"
    Write-Host "  pip install -U huggingface_hub; hf auth login"
    Write-Host "Then run START-HERE.bat again. It continues the download."
} elseif ($todo.Count -and -not $tok) {
    Write-Host 'Tip: "hf auth login" (a free Hugging Face account) gives the download higher rate limits.'
}
if ($todo.Count -and $hf) {
    Write-Host "Downloading with hf, $jobs files at once:"
    & hf download $repo @($todo | ForEach-Object { $_.path }) --revision $rev --local-dir $dir --max-workers $jobs
    if ($LASTEXITCODE) { Write-Host "hf stopped. curl downloads the files that remain." }
    Write-Host "Checking the files (SHA-256)..."
    foreach ($f in $todo) {
        $dst = Join-Path $dir $f.path
        if (Good $dst $f) {
            Add-Content $okf (Line $f)
            if (Test-Path "$dst.part") { Remove-Item "$dst.part" }
        } elseif (Test-Path $dst) { Remove-Item $dst }
    }
    $todo = @(Todo)
}
function Start-Curl($f) {   # curl into FILE.part (a stalled connection resumes it). $null: FILE.part is complete
    $dst = Join-Path $dir $f.path
    New-Item -ItemType Directory -Force (Split-Path $dst) | Out-Null
    if ((Test-Path "$dst.part") -and (Get-Item "$dst.part").Length -ge $f.size) {
        if (Good "$dst.part" $f) { return $null }
        Remove-Item "$dst.part"
    }
    $a = '-fsSL --retry 20 --speed-limit 100000 --speed-time 60 -C - -H "@{0}" -o "{1}.part" "https://huggingface.co/{2}/resolve/{3}/{4}"' -f $hfile, $dst, $repo, $rev, $f.path
    Start-Process curl.exe -ArgumentList $a -NoNewWindow -PassThru
}
function Finish($f) {   # FILE.part becomes FILE if it checks out
    $dst = Join-Path $dir $f.path
    if (-not (Test-Path "$dst.part")) { Write-Host "The download of $($f.path) stopped."; return }
    if ((Get-Item "$dst.part").Length -lt $f.size) { Write-Host "$($f.path) is not complete."; return }   # the next run resumes it
    if (-not (Good "$dst.part" $f)) { Remove-Item "$dst.part"; Write-Host "$($f.path) failed its size or SHA-256 check and was removed."; return }
    Move-Item -Force "$dst.part" $dst
    Add-Content $okf (Line $f)
    Write-Host $f.path
}
function Got {   # the bytes of the files in $todo: FILE, else FILE.part
    $b = 0
    foreach ($f in $todo) {
        $dst = Join-Path $dir $f.path
        if (Test-Path $dst) { $b += $f.size } elseif (Test-Path "$dst.part") { $b += (Get-Item "$dst.part").Length }
    }
    $b
}
if ($todo.Count) {
    Write-Host "Downloading $($todo.Count) files with curl, $jobs at once:"
    $run = @{}   # path -> @(file, curl process or $null)
    $next, $b0, $t0, $shown = 0, (Got), (Get-Date), (Get-Date)
    $all = 0
    foreach ($f in $todo) { $all += $f.size }
    while ($next -lt $todo.Count -or $run.Count) {
        foreach ($k in @($run.Keys)) {
            $p = $run[$k][1]
            if (-not $p -or $p.HasExited) { Finish $run[$k][0]; $run.Remove($k) }
        }
        while ($next -lt $todo.Count -and $run.Count -lt $jobs) {
            $f = $todo[$next]; $next += 1
            $run[$f.path] = @($f, (Start-Curl $f))
        }
        if (((Get-Date) - $shown).TotalSeconds -ge 30) {   # a progress line every 30 s
            $shown = Get-Date
            $b = (Got) - $b0; $s = ((Get-Date) - $t0).TotalSeconds
            Write-Host ("{0:N1} of {1:N1} GB, {2:N0} MB/s" -f (((Got)) / 1e9), ($all / 1e9), ($b / [math]::Max($s, 1) / 1e6))
        }
        Start-Sleep -Milliseconds 500
    }
    $todo = @(Todo)
    if ($todo.Count) { Fail "$($todo.Count) files did not download or check out (see above). Run START-HERE.bat again to resume." }
}
Remove-Item -Force -ErrorAction SilentlyContinue $hfile
$hfile = $null
$pline = 'rem the pack''s own 8-bit PLE table (set Q4E_PLE_GGUF=FILE reads another, as a GGUF)'

Write-Host "`nThe pack is ready. One short chat:"
& build\bin\nslm-chat.exe --model $dir --no-think --max-tokens 64 "Say hello in one short sentence."
if ($LASTEXITCODE) { Fail "nslm-chat failed. See the message above." }
$run = Join-Path $here "run-$pick.bat"
Set-Content -Encoding ascii $run @(
    '@echo off', 'cd /d "%~dp0"',
    "echo NanoSeedLM $pick`: an OpenAI-compatible API on http://127.0.0.1:8080/v1 (Ctrl+C stops it)", $pline,
    "build\bin\nslm-serve.exe --model `"$dir`" --ctx 32768 --port 8080", 'pause')
Write-Host "`nWrote $run. Next time, run it to start the server at once."
& cmd /c $run
