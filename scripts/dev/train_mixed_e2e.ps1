param(
  [string]$TrainDirs = "data\train\funsd\hf_train,data\train\ph_forms\annotations",
  [string]$ValDir    = "data\val\funsd\hf_test",
  [string]$Labels    = "data\labels_union.json",
  [string]$SaveDir   = "saved_models\mixed_funsd_ph_e6",
  [int]$Epochs = 6
)

python -m scripts.build_union_labels `
  --labels $Labels `
  --scan_dirs "$TrainDirs,$ValDir"

python -m scripts.dataset_audit `
  --dirs "data\train\ph_forms\annotations,data\val\ph_forms\annotations,data\test\ph_forms\annotations" `
  --labels_map $Labels

python -m scripts.train_classifier_multi `
  --train_dirs $TrainDirs `
  --val_dir    $ValDir `
  --labels     $Labels `
  --epochs     $Epochs `
  --batch_size 4 `
  --num_workers 0 `
  --balance_datasets `
  --seed 42 `
  --save_dir   $SaveDir

python -m scripts.evaluate_test `
  --test_dir   data\val\funsd\hf_test `
  --labels     $Labels `
  --checkpoint $SaveDir\classifier.pt `
  --exclude_o `
  --report_txt static\metrics_funsd.txt

python -m scripts.evaluate_test `
  --test_dir   data\test\ph_forms\annotations `
  --labels     $Labels `
  --checkpoint $SaveDir\classifier.pt `
  --exclude_o `
  --report_txt static\metrics_ph.txt
