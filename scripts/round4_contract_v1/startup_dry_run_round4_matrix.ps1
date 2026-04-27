param(
    [string]$Python = "E:\anaconda3\envs\tdc_env\python.exe"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Workspace = Resolve-Path (Join-Path $ScriptDir "..\..")
$RunRepro = Join-Path $Workspace "run_repro.py"
$ConfigDir = Join-Path $Workspace "configs\round4_contract_v1"
$Seeds = @(20260414, 20260417, 20260418)
$Checks = @(
    @{ Config = "p003_baseline.json"; Setting = "exp4" },
    @{ Config = "p003_baseline.json"; Setting = "exp5" },
    @{ Config = "p003_baseline.json"; Setting = "exp6" },
    @{ Config = "full_lora_rank8.json"; Setting = "exp4" },
    @{ Config = "full_lora_rank8.json"; Setting = "exp5" },
    @{ Config = "full_lora_rank8.json"; Setting = "exp6" },
    @{ Config = "frozen_no_adapter.json"; Setting = "exp5" },
    @{ Config = "frozen_no_adapter.json"; Setting = "exp6" },
    @{ Config = "full_finetune.json"; Setting = "exp5" },
    @{ Config = "full_finetune.json"; Setting = "exp6" },
    @{ Config = "lora_rank2.json"; Setting = "exp5" },
    @{ Config = "lora_rank2.json"; Setting = "exp6" },
    @{ Config = "lora_rank4.json"; Setting = "exp5" },
    @{ Config = "lora_rank4.json"; Setting = "exp6" },
    @{ Config = "lora_rank8.json"; Setting = "exp5" },
    @{ Config = "lora_rank8.json"; Setting = "exp6" },
    @{ Config = "lora_rank16.json"; Setting = "exp5" },
    @{ Config = "lora_rank16.json"; Setting = "exp6" },
    @{ Config = "no_preservation_lock_rank8.json"; Setting = "exp5" },
    @{ Config = "no_preservation_lock_rank8.json"; Setting = "exp6" },
    @{ Config = "merged_lora_rank8.json"; Setting = "exp5" },
    @{ Config = "merged_lora_rank8.json"; Setting = "exp6" },
    @{ Config = "unmerged_lora_rank8.json"; Setting = "exp5" },
    @{ Config = "unmerged_lora_rank8.json"; Setting = "exp6" }
)

foreach ($Check in $Checks) {
    foreach ($Seed in $Seeds) {
        $ConfigPath = Join-Path $ConfigDir $Check.Config
        $RunId = "startup_dry_run_{0}_{1}_seed{2}" -f ([IO.Path]::GetFileNameWithoutExtension($Check.Config)), $Check.Setting, $Seed
        & $Python $RunRepro `
            --round4-config $ConfigPath `
            --experiments $Check.Setting `
            --seed $Seed `
            --run-id $RunId `
            --startup-dry-run
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
    }
}
