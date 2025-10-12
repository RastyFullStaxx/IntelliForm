param(
  [switch]$Prod # add -Prod to write into explanations\... (default uses _staging)
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# --- Paths (staging vs prod) ---
$repoRoot  = (Resolve-Path ".").Path
$explBase  = Join-Path $repoRoot "explanations"
if (-not $Prod) { $explBase = Join-Path $explBase "_staging" }

$trainDir  = Join-Path $explBase "training"
$logsDir   = Join-Path $explBase "logs"
$toolLog   = Join-Path $logsDir "tool_metrics.jsonl"
$backupDir = Join-Path $logsDir "_backups"
$regJsonl  = Join-Path $explBase "registry.jsonl"

# Ensure dirs
New-Item -ItemType Directory -Force -Path $trainDir,$logsDir,$backupDir | Out-Null
if (-not (Test-Path $regJsonl)) {
  Write-Host "Registry not found at $regJsonl. We'll fall back to scanning explainer JSONs by title." -ForegroundColor Yellow
}

function ToIsoUtc([string]$dtString) {
  try {
    $dt = [DateTime]::Parse($dtString, [System.Globalization.CultureInfo]::InvariantCulture)
    return $dt.ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
  } catch {
    return (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
  }
}

# --- Desired metric rows (your list) ---
$desired = @(
  @{ Title="PASSPORT APPLICATION FORM";                         TS="10/13/2025, 6:12:36 AM"; TP=30; FP=0; FN=0; P=0.9; R=0.9; F1=0.9 },
  @{ Title="Government PassportApplicationForm";                TS="10/13/2025, 6:12:35 AM"; TP=30; FP=0; FN=0; P=0.9; R=0.9; F1=0.9 },

  # BIR Treaty Purposes — map via filename fragments
  @{ FileContains="0901-S1"; Title="Application for Treaty Purposes"; TS="10/13/2025, 6:06:06 AM"; TP=23; FP=0; FN=0; P=0.9; R=0.9; F1=0.9 },
  @{ FileContains="0901-S2"; Title="Application for Treaty Purposes"; TS="10/13/2025, 6:06:07 AM"; TP=24; FP=0; FN=0; P=0.9; R=0.9; F1=0.9 },
  @{ FileContains="0901-S3"; Title="Application for Treaty Purposes"; TS="10/13/2025, 6:06:07 AM"; TP=25; FP=0; FN=0; P=0.9; R=0.9; F1=0.9 },
  @{ FileContains="0901-S4"; Title="Application for Treaty Purposes"; TS="10/13/2025, 6:06:07 AM"; TP=30; FP=0; FN=0; P=0.9; R=0.9; F1=0.9 },
  @{ FileContains="0901-T";  Title="Application for Treaty Purposes"; TS="10/13/2025, 6:06:07 AM"; TP=30; FP=0; FN=0; P=0.9; R=0.9; F1=0.9 },

  @{ Title="Finance Auto InvestEnrollmentForm";                  TS="10/13/2025, 6:06:07 AM"; TP=10; FP=0; FN=0; P=0.9; R=0.9; F1=0.9; FileContains="Auto-InvestEnrollmentForm" },
  @{ Title="Government NBIClearanceApplicationForm";             TS="10/13/2025, 6:06:07 AM"; TP=40; FP=0; FN=0; P=0.9; R=0.9; F1=0.9; FileContains="NBIClearance" },
  @{ Title="Finance Suitability Assessment Form";                TS="10/13/2025, 6:06:07 AM"; TP=25; FP=0; FN=0; P=0.9; R=0.9; F1=0.9; FileContains="SuitabilityAssessment" },
  @{ Title="APPLICATION FOR HEALTH CARE ASSISTANCE";             TS="10/13/2025, 6:06:07 AM"; TP=15; FP=0; FN=0; P=0.9; R=0.9; F1=0.9; FileContains="HealthcareApplicationForm" },
  @{ Title="EAZY HEALTH CLAIM FORM FOR CRITICAL ILLNESS & DISABILITY"; TS="10/13/2025, 6:06:07 AM"; TP=20; FP=0; FN=0; P=0.9; R=0.9; F1=0.9; FileContains="EAZyHealthClaimForm" },
  @{ Title="AutoCharge Facility (ACF) for BPI Credit Cards";     TS="10/13/2025, 6:06:07 AM"; TP=20; FP=0; FN=0; P=0.9; R=0.9; F1=0.9; FileContains="CreditCardholdersAutoChargeForm" },
  @{ Title="Government BusinessPermitApplicationForm";           TS="10/13/2025, 6:06:07 AM"; TP=40; FP=0; FN=0; P=0.9; R=0.9; F1=0.9; FileContains="BusinessPermitApplicationForm" },
  @{ Title="Government UnifiedMulti PurposeIDApplication Form";  TS="10/13/2025, 6:06:07 AM"; TP=30; FP=0; FN=0; P=0.9; R=0.9; F1=0.9; FileContains="UnifiedMulti-PurposeID" },
  @{ Title="OPTIONAL LIFE INSURANCE POLICY LOAN APPLICATION";    TS="10/13/2025, 6:06:06 AM"; TP=20; FP=0; FN=0; P=0.9; R=0.9; F1=0.9 },
  @{ Title="Prosperity Card Purchase Form";                      TS="10/13/2025, 6:06:06 AM"; TP=27; FP=0; FN=0; P=0.9; R=0.9; F1=0.9; FileContains="ProsperityCardPurchase" },
  @{ Title="Personal Accident Claim Form";                       TS="10/13/2025, 6:06:06 AM"; TP=20; FP=0; FN=0; P=0.9; R=0.9; F1=0.9; FileContains="PersonalAccidentClaim" },
  @{ Title="Healthcare Insurance Sun Life Variable Life Insurance Request for Fund Withdrawal"; TS="10/13/2025, 6:06:06 AM"; TP=3; FP=0; FN=0; P=0.9; R=0.9; F1=0.9; FileContains="VariableLifeInsuranceRequestforFundWithdrawal" }
)

# --- Load registry.jsonl -> map canonical_id => expl_json path & source_pdf ---
$byCanon = @{}
if (Test-Path $regJsonl) {
  Get-Content $regJsonl | ForEach-Object {
    if (-not $_) { return }
    try {
      $row = $_ | ConvertFrom-Json
      $cid = $row.canonical_id
      if ($cid) {
        $byCanon[$cid] = @{
          expl_json  = (Join-Path $repoRoot ($row.expl_json -replace '[\\/]+','\'))
          source_pdf = $row.source_pdf
          title      = $row.title
          bucket     = $row.bucket
        }
      }
    } catch { }
  }
}

function HasKey($h, [string]$k) {
  return ($h -is [hashtable]) -and $h.ContainsKey($k) -and ($null -ne $h[$k]) -and ($h[$k].ToString().Trim() -ne "")
}

# Helper to find explainer path for a desired item
function Find-ExplPath($d){
  # 1) Try registry by FileContains (source_pdf contains fragment)
  if (HasKey $d 'FileContains') {
    $frag = $d['FileContains']
    foreach($cid in $byCanon.Keys){
      $meta = $byCanon[$cid]
      if ($meta.source_pdf -and ($meta.source_pdf -like "*$frag*")) {
        return @{ Path=$meta.expl_json; Canon=$cid }
      }
    }
  }
  # 2) Fallback: title match inside explainer JSONs
  $files = Get-ChildItem $trainDir -Filter *.json -File -ErrorAction SilentlyContinue
  foreach($f in $files){
    try {
      $j = Get-Content $f.FullName -Raw | ConvertFrom-Json
      $t = [string]$j.title
      if ($t -and $d.Title -and ($t.Trim().ToLower() -eq $d.Title.Trim().ToLower())) {
        return @{ Path=$f.FullName; Canon=($j.canonical_id) }
      }
    } catch { }
  }
  return $null
}

# Backup tool_metrics.jsonl
if (Test-Path $toolLog) {
  $stamp = (Get-Date).ToString("yyyyMMdd-HHmmss")
  Copy-Item $toolLog (Join-Path $backupDir "tool_metrics-$stamp.jsonl") -Force
}

# --- Patch loop ---
$written = 0
$missing = @()

foreach($d in $desired){
  $hit = Find-ExplPath $d
  if (-not $hit) {
    $missing += $d
    continue
  }
  $exPath = $hit.Path
  $cid    = $hit.Canon
  if (-not (Test-Path $exPath)) { $missing += $d; continue }

  $iso = ToIsoUtc $d.TS

  # Load, patch, save explainer JSON
  $json = Get-Content $exPath -Raw | ConvertFrom-Json
  if (-not $json.metrics) { $json | Add-Member -Name metrics -MemberType NoteProperty -Value @{} }
  $json.metrics.tp        = [int]$d.TP
  $json.metrics.fp        = [int]$d.FP
  $json.metrics.fn        = [int]$d.FN
  $json.metrics.precision = [double]$d.P
  $json.metrics.recall    = [double]$d.R
  $json.metrics.f1        = [double]$d.F1
  $json.updated_at        = $iso

  ($json | ConvertTo-Json -Depth 50) | Set-Content -Encoding UTF8 $exPath

  # Append a tool-metrics row for dashboard
  $row = [ordered]@{
    row_id      = [guid]::NewGuid().ToString()
    ts_utc      = $iso
    canonical_id= $json.canonical_id
    form_title  = $json.title
    bucket      = "training"
    source      = "training"
    metrics     = @{
      tp=$d.TP; fp=$d.FP; fn=$d.FN; precision=$d.P; recall=$d.R; f1=$d.F1
    }
    note        = "Patched metrics (dashboard backfill)"
  }
  ($row | ConvertTo-Json -Depth 20) | Out-File -FilePath $toolLog -Append -Encoding utf8
  Write-Host "Updated: $($json.title)  ->  $([IO.Path]::GetFileName($exPath))" -ForegroundColor Green
  $written++
}

Write-Host "`nDone. Patched $written item(s)." -ForegroundColor Cyan

if ($missing.Count -gt 0) {
  Write-Host "`nCould not locate explainer for:" -ForegroundColor Yellow
  foreach ($m in $missing) {
    $hint = ""
    if (HasKey $m 'FileContains') {
      $hint = " (FileContains: $($m['FileContains']))"
    }
    Write-Host (" - {0}{1}" -f $m.Title, $hint)
  }
}

