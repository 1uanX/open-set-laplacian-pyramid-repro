param(
    [string]$Python = "E:\anaconda3\envs\tdc_env\python.exe",
    [ValidateSet("all", "primary", "ablations")]
    [string]$Matrix = "all"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Workspace = Resolve-Path (Join-Path $ScriptDir "..\..")
$RunRepro = Join-Path $Workspace "run_repro.py"
$ConfigDir = Join-Path $Workspace "configs\round4_contract_v1"
$BatchId = "round4_v1_matrix_{0}" -f (Get-Date -Format "yyyyMMddHHmmss")
$ManifestDir = Join-Path $Workspace "output\round4_contract_v1\script_manifests"
New-Item -ItemType Directory -Force -Path $ManifestDir | Out-Null

$Seeds = @(20260414, 20260417, 20260418)

$Rows = @(
    @{ Group = "primary"; Stem = "p003_baseline"; Config = "p003_baseline.json"; Settings = @("exp4", "exp5", "exp6") },
    @{ Group = "primary"; Stem = "full_lora_rank8"; Config = "full_lora_rank8.json"; Settings = @("exp4", "exp5", "exp6") },
    @{ Group = "ablations"; Stem = "frozen_no_adapter"; Config = "frozen_no_adapter.json"; Settings = @("exp5", "exp6") },
    @{ Group = "ablations"; Stem = "full_finetune"; Config = "full_finetune.json"; Settings = @("exp5", "exp6") },
    @{ Group = "ablations"; Stem = "lora_rank2"; Config = "lora_rank2.json"; Settings = @("exp5", "exp6") },
    @{ Group = "ablations"; Stem = "lora_rank4"; Config = "lora_rank4.json"; Settings = @("exp5", "exp6") },
    @{ Group = "ablations"; Stem = "lora_rank8"; Config = "lora_rank8.json"; Settings = @("exp5", "exp6") },
    @{ Group = "ablations"; Stem = "lora_rank16"; Config = "lora_rank16.json"; Settings = @("exp5", "exp6") },
    @{ Group = "ablations"; Stem = "no_preservation_lock_rank8"; Config = "no_preservation_lock_rank8.json"; Settings = @("exp5", "exp6") },
    @{ Group = "ablations"; Stem = "merged_lora_rank8"; Config = "merged_lora_rank8.json"; Settings = @("exp5", "exp6") },
    @{ Group = "ablations"; Stem = "unmerged_lora_rank8"; Config = "unmerged_lora_rank8.json"; Settings = @("exp5", "exp6") }
)

if ($Matrix -ne "all") {
    $Rows = @($Rows | Where-Object { $_.Group -eq $Matrix })
}

$Planned = @()
foreach ($Row in $Rows) {
    foreach ($Setting in $Row.Settings) {
        foreach ($Seed in $Seeds) {
            $RunId = "round4_v1_{0}_{1}_seed{2}_{3}" -f $Row.Stem, $Setting, $Seed, $BatchId
            $Planned += [ordered]@{
                batch_id = $BatchId
                group = $Row.Group
                run_id = $RunId
                config = $Row.Config
                setting = $Setting
                seed = $Seed
                contract_version = "round4_v1"
                status = "planned"
                primary_open_set_anchor = "P003"
                r4_b02_role = "secondary_reference_only"
                deployment_diagnostics = @("trainable_parameters", "peak_memory", "inference_latency")
            }
        }
    }
}

$PlanPath = Join-Path $ManifestDir ("{0}_planned.json" -f $BatchId)
$Planned | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $PlanPath -Encoding UTF8

$Failures = @()
foreach ($Item in $Planned) {
    $ConfigPath = Join-Path $ConfigDir $Item.config
    $OutputDir = Join-Path $Workspace ("output\round4_contract_v1\{0}" -f $Item.run_id)
    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

    $PreStatus = [ordered]@{
        batch_id = $BatchId
        run_id = $Item.run_id
        config_path = $ConfigPath
        setting = $Item.setting
        seed = $Item.seed
        status = "started_by_script"
        completion_status = "incomplete_until_run_repro_finishes"
        started_at = (Get-Date).ToUniversalTime().ToString("o")
        planned_manifest = $PlanPath
    }
    $PreStatus | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $OutputDir "script_status.pre.json") -Encoding UTF8

    $StdoutPath = Join-Path $OutputDir "run_stdout.log"
    $StderrPath = Join-Path $OutputDir "run_stderr.log"
    $CommandArgs = @(
        $RunRepro,
        "--round4-config", $ConfigPath,
        "--experiments", $Item.setting,
        "--seed", $Item.seed,
        "--run-id", $Item.run_id,
        "--output-dir", $OutputDir
    )

    & $Python @CommandArgs 1> $StdoutPath 2> $StderrPath
    $ExitCode = $LASTEXITCODE
    if ($ExitCode -eq 0) {
        $PostStatus = $PreStatus.Clone()
        $PostStatus.status = "completed_by_script"
        $PostStatus.completion_status = "completed"
        $PostStatus.completed_at = (Get-Date).ToUniversalTime().ToString("o")
        $PostStatus.exit_code = $ExitCode
        $PostStatus | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $OutputDir "script_status.completed.json") -Encoding UTF8
    } else {
        $PostStatus = $PreStatus.Clone()
        $PostStatus.status = "failed_by_script"
        $PostStatus.completion_status = "failed"
        $PostStatus.completed_at = (Get-Date).ToUniversalTime().ToString("o")
        $PostStatus.exit_code = $ExitCode
        $PostStatus.stdout_path = $StdoutPath
        $PostStatus.stderr_path = $StderrPath
        $PostStatus | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $OutputDir "script_status.failed.json") -Encoding UTF8
        $Failures += $Item.run_id
    }
}

$Summary = [ordered]@{
    batch_id = $BatchId
    matrix = $Matrix
    planned_count = $Planned.Count
    failed_count = $Failures.Count
    failed_run_ids = $Failures
    status = $(if ($Failures.Count -eq 0) { "completed" } else { "completed_with_failures_visible" })
}
$SummaryPath = Join-Path $ManifestDir ("{0}_summary.json" -f $BatchId)
$Summary | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $SummaryPath -Encoding UTF8

if ($Failures.Count -gt 0) {
    exit 1
}
exit 0
