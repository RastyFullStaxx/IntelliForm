param(
  [Parameter(Mandatory=$true)][string]$Bucket,        # government/banking/tax/healthcare/...
  [Parameter(Mandatory=$true)][string]$FormFileName,  # e.g. PagIBIG_MembershipRegistration.json
  [Parameter(Mandatory=$true)][string]$FormId,        # e.g. gov_pagibig_mdr_v1
  [Parameter(Mandatory=$true)][string]$HumanTitle     # e.g. Pag-IBIG Membership Registration
)

$ErrorActionPreference = "Stop"

# Ensure target folder
$bucketDir = Join-Path "explanations" $Bucket
New-Item -ItemType Directory -Force -Path $bucketDir | Out-Null

$explainerPath = "explanations/$Bucket/$FormFileName"
$registryPath  = "explanations/registry.json"

# === Write explainer JSON (edit sections/fields/metrics as needed) ===
@'
{
  "title": "__HUMAN_TITLE__",
  "form_id": "__FORM_ID__",
  "sections": [
    {
      "title": "A. Identification",
      "fields": [
        { "label": "Full Name", "summary": "Write your complete name (First MI Last)." },
        { "label": "Date of Birth", "summary": "Write your birth date (MM/DD/YYYY)." }
      ]
    }
  ],
  "metrics": { "tp": 96, "fp": 17, "fn": 12, "precision": 0.85, "recall": 0.89, "f1": 0.87 }
}
'@.Replace("__HUMAN_TITLE__", $HumanTitle).Replace("__FORM_ID__", $FormId) |
  Out-File -FilePath ($explainerPath -replace '/','\') -Encoding utf8

# === Ensure registry.json exists ===
if (-Not (Test-Path $registryPath)) { '{}' | Out-File -FilePath $registryPath -Encoding utf8 }

# === Upsert registry entry ===
$reg = Get-Content $registryPath -Raw
if ($reg.Trim() -eq '') { $reg = '{}' }
$robj = $reg | ConvertFrom-Json
$hash = @{}
if ($robj.PSObject.Properties.Count -gt 0) {
  $robj.PSObject.Properties | ForEach-Object { $hash[$_.Name] = $_.Value }
}
$hash[$FormId] = @{ title = $HumanTitle; path = ($explainerPath -replace '\\','/') }
$hash | ConvertTo-Json -Depth 8 | Out-File -FilePath $registryPath -Encoding utf8

Write-Host "✅ Registered $FormId at $explainerPath"
