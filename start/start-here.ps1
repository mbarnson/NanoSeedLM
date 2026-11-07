# start/start-here.ps1 [PACK [MODEL_FOLDER]] - NanoSeedLM's start on a Windows PC (START-HERE.bat runs it): check the PC,
# pick a pack, download and check it, chat once, write run-PACK.bat and start the server. Run it again to resume.
param([string]$Pack, [string]$ModelDir)   # given: no questions
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false   # PowerShell 7: the script reads curl's and nslm-chat's exit codes
$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here
function Fail($m) { Write-Host "`n$m" -ForegroundColor Red; exit 1 }

# name = repository, revision, least RAM GB, least VRAM GB, download GB (the model cards' "For" column)
$packs = [ordered]@{
    'TCQ4'    = @('txgsync/Qwen3.8-Flash-Next-NanoSeedLM-TCQ4', '4e0d254480f24d444082e935c9b182a4e582730f', 96, 24, 128)
    'TCQ3-D4' = @('txgsync/Qwen3.8-Flash-Next-NanoSeedLM-TCQ3-D4', 'e29730a3fea357b710f0b5bd57158c74a82cd7f1', 96, 16, 117)
    'TCQ3'    = @('txgsync/Qwen3.8-Flash-Next-NanoSeedLM-TCQ3', 'a850ead9cc4e852b9566aaec851213f7e813d193', 64, 12, 112)
}

Write-Host "NanoSeedLM: Qwen3.8-Flash-Next on this PC`n"
if (-not (Test-Path build\bin\nslm-serve.exe)) { Fail "build\bin\nslm-serve.exe is missing. Unpack the whole engine zip, then run START-HERE.bat in it." }
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { Fail "No NVIDIA driver found. Install driver 570 or newer: https://www.nvidia.com/drivers" }
$g = ((& nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits) | Select-Object -First 1).Split(',') | ForEach-Object { $_.Trim() }
$vram = [math]::Round([double]$g[1] / 1024)
$ram = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
Write-Host "GPU: $($g[0]), $vram GB, driver $($g[2]). RAM: $ram GB."
if ([int]$g[2].Split('.')[0] -lt 570) { Fail "The driver is older than 570 (CUDA 12.8). Update it: https://www.nvidia.com/drivers" }
if ($vram -lt 12) { Write-Host "Warning: less than 12 GB of VRAM. The engine may not fit." -ForegroundColor Yellow }

function Fits($have, $size) { $have -ge 0.9 * $size }   # a 64 GB PC may report 62 GB: 90% of a size counts as that size
$rec = $null
foreach ($n in $packs.Keys) { $p = $packs[$n]; if (-not $rec -and (Fits $ram $p[2]) -and (Fits $vram $p[3])) { $rec = $n } }
if (-not $rec) { if (-not (Fits $ram 64)) { Fail "The smallest pack (TCQ3) needs 64 GB of RAM. This PC has $ram GB." }; $rec = 'TCQ3' }
Write-Host "`nPacks (KL against BF16: lower is closer to the original model):"
Write-Host "  TCQ4     0.0415  for a GPU of 24 GB or more with 96 to 128 GB of RAM ($($packs['TCQ4'][4]) GB download)"
Write-Host "  TCQ3-D4  0.0638  for a 16 GB GPU with 96 to 128 GB of RAM ($($packs['TCQ3-D4'][4]) GB download)"
Write-Host "  TCQ3     0.0778  for a 12 or 16 GB GPU with 64 GB of RAM ($($packs['TCQ3'][4]) GB download)"
Write-Host "Each pack carries its PLE table at 8 bits. It stays on the disk, and the page cache holds the rows in use."
$pick = $Pack
if (-not $pick) { $pick = Read-Host "Pack [$rec]" }
if (-not $pick) { $pick = $rec }
if (-not $packs.Contains($pick)) { Fail "Unknown pack $pick. Use TCQ4, TCQ3-D4 or TCQ3." }
$root = $ModelDir
if (-not $root -and -not $Pack) { $root = Read-Host "Model folder [$here\models]" }
if (-not $root) { $root = "$here\models" }
$repo, $rev = $packs[$pick][0], $packs[$pick][1]
$dir = Join-Path $root ($repo.Split('/')[1])
New-Item -ItemType Directory -Force $dir | Out-Null

# the files at their pinned revisions, each checked by size and SHA-256 (verified ones are listed in .nslm-verified)
$hdr = @{}
$auth = @()
if ($env:HF_TOKEN) { $hdr['Authorization'] = "Bearer $env:HF_TOKEN"; $auth = @('-H', "Authorization: Bearer $env:HF_TOKEN") }
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
    if ($ok.ContainsKey((Line $f))) { $new.Add((Line $f)); continue }
    $dst = Join-Path $dir $f.path
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
$ok = @{}
foreach ($l in $new) { $ok[$l] = 1 }
$i = 0
foreach ($f in $files) {
    $i += 1
    if ($ok.ContainsKey((Line $f))) { continue }
    $dst = Join-Path $dir $f.path
    New-Item -ItemType Directory -Force (Split-Path $dst) | Out-Null
    $have = 0
    if (Test-Path "$dst.part") { $have = (Get-Item "$dst.part").Length }
    Write-Host ("[{0}/{1}] {2} ({3})" -f $i, $files.Count, $f.path, $(if ($f.size -lt 1e9) { "{0:N0} MB" -f ($f.size / 1e6) } else { "{0:N1} GB" -f ($f.size / 1e9) }))
    if ($have -ne $f.size) {   # into FILE.part: a stalled connection resumes it, and FILE appears once it checks out
        & curl.exe -fL --retry 20 --speed-limit 100000 --speed-time 60 -C - --progress-bar @auth -o "$dst.part" "https://huggingface.co/$repo/resolve/$rev/$($f.path)"
        if ($LASTEXITCODE) { Fail "The download of $($f.path) stopped (curl $LASTEXITCODE). Run START-HERE.bat again to resume." }
    }
    if ((Get-Item "$dst.part").Length -ne $f.size) { Fail "$($f.path) has the wrong size. Run START-HERE.bat again." }
    if ($f.sha -and (Get-FileHash -Algorithm SHA256 "$dst.part").Hash.ToLower() -ne $f.sha) {
        Remove-Item "$dst.part"
        Fail "$($f.path) failed its SHA-256 check and was removed. Run START-HERE.bat again."
    }
    Move-Item -Force "$dst.part" $dst
    Add-Content $okf (Line $f)
}
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
