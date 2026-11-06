# start/start-here.ps1 - NanoSeedLM's start on a Windows PC (START-HERE.bat runs it from the engine folder): check the PC,
# pick a pack, download and check it, chat once, write run-PACK.bat and start the server. Run it again to resume.
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here
function Fail($m) { Write-Host "`n$m" -ForegroundColor Red; exit 1 }

# name = repository, revision, least RAM GB, least VRAM GB, download GB (the model cards' "For" column)
$packs = [ordered]@{
    'TQ4'    = @('txgsync/Qwen3.8-Flash-Next-NanoSeedLM-TQ4', 'main', 96, 24, 102)
    'TQ3-D4' = @('txgsync/Qwen3.8-Flash-Next-NanoSeedLM-TQ3-D4', 'main', 96, 16, 92)
    'TQ3'    = @('txgsync/Qwen3.8-Flash-Next-NanoSeedLM-TQ3', 'main', 64, 16, 87)
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

$rec = $null
foreach ($n in $packs.Keys) { $p = $packs[$n]; if (-not $rec -and $ram -ge $p[2] -and $vram -ge $p[3]) { $rec = $n } }
if (-not $rec) { if ($ram -lt 64) { Fail "The smallest pack (TQ3) needs 64 GB of RAM. This PC has $ram GB." }; $rec = 'TQ3' }
Write-Host "`nPacks (KL against BF16: lower is closer to the original model):"
Write-Host "  TQ4     0.0415  for a GPU of 24 GB or more with 96 to 128 GB of RAM ($($packs['TQ4'][4]) GB download)"
Write-Host "  TQ3-D4  0.0638  for a 16 GB GPU with 96 to 128 GB of RAM ($($packs['TQ3-D4'][4]) GB download)"
Write-Host "  TQ3     0.0778  for a 16 GB GPU with 64 GB of RAM ($($packs['TQ3'][4]) GB download)"
$pick = Read-Host "Pack [$rec]"
if (-not $pick) { $pick = $rec }
if (-not $packs.Contains($pick)) { Fail "Unknown pack $pick. Use TQ4, TQ3-D4 or TQ3." }
$root = Read-Host "Model folder [$here\models]"
if (-not $root) { $root = "$here\models" }
$repo, $rev = $packs[$pick][0], $packs[$pick][1]
$dir = Join-Path $root ($repo.Split('/')[1])
New-Item -ItemType Directory -Force $dir | Out-Null

# the pack's files at the pinned revision, each checked by size and SHA-256 (verified ones are listed in .nslm-verified)
$hdr = @{}
$auth = @()
if ($env:HF_TOKEN) { $hdr['Authorization'] = "Bearer $env:HF_TOKEN"; $auth = @('-H', "Authorization: Bearer $env:HF_TOKEN") }
$tree = Invoke-RestMethod -Headers $hdr "https://huggingface.co/api/models/$repo/tree/$($rev)?recursive=true"   # PowerShell 5.1: one array object
$files = @($tree | Where-Object { $_.type -eq 'file' -and -not $_.path.StartsWith('.') })
$okf = Join-Path $dir '.nslm-verified'
$ok = @{}
if (Test-Path $okf) { Get-Content $okf | ForEach-Object { $ok[$_] = 1 } }
$need = 0
foreach ($f in $files) { if (-not $ok.ContainsKey($f.path)) { $need += $f.size } }
$free = (Get-Item $dir).PSDrive.Free
if ($need -gt $free) { Fail ("The download needs {0:N0} GB more and the disk has {1:N0} GB free." -f ($need / 1GB), ($free / 1GB)) }
Write-Host ("`nDownloading {0} ({1:N0} GB to go) into {2}" -f $repo, ($need / 1GB), $dir)
$i = 0
foreach ($f in $files) {
    $i += 1
    if ($ok.ContainsKey($f.path)) { continue }
    $dst = Join-Path $dir $f.path
    New-Item -ItemType Directory -Force (Split-Path $dst) | Out-Null
    $have = 0
    if (Test-Path $dst) { $have = (Get-Item $dst).Length }
    Write-Host ("[{0}/{1}] {2} ({3:N1} GB)" -f $i, $files.Count, $f.path, ($f.size / 1GB))
    if ($have -ne $f.size) {
        & curl.exe -fL --retry 5 -C - --progress-bar @auth -o $dst "https://huggingface.co/$repo/resolve/$rev/$($f.path)"
        if ($LASTEXITCODE) { Fail "The download of $($f.path) stopped (curl $LASTEXITCODE). Run START-HERE.bat again to resume." }
    }
    if ((Get-Item $dst).Length -ne $f.size) { Fail "$($f.path) has the wrong size. Run START-HERE.bat again." }
    if ($f.lfs -and (Get-FileHash -Algorithm SHA256 $dst).Hash.ToLower() -ne $f.lfs.oid) {
        Remove-Item $dst
        Fail "$($f.path) failed its SHA-256 check and was removed. Run START-HERE.bat again."
    }
    Add-Content $okf $f.path
}

Write-Host "`nThe pack is ready. One short chat:"
& build\bin\nslm-chat.exe --model $dir --no-think --max-tokens 64 "Say hello in one short sentence."
if ($LASTEXITCODE) { Fail "nslm-chat failed. See the message above." }
$run = Join-Path $here "run-$pick.bat"
Set-Content -Encoding ascii $run @(
    '@echo off', 'cd /d "%~dp0"',
    "echo NanoSeedLM $pick`: an OpenAI-compatible API on http://127.0.0.1:8080/v1 (Ctrl+C stops it)",
    "build\bin\nslm-serve.exe --model `"$dir`" --ctx 32768 --port 8080", 'pause')
Write-Host "`nWrote $run. Next time, run it to start the server at once."
& cmd /c $run
