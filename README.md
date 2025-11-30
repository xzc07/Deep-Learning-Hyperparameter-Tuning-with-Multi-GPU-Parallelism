# ResNet-50 Fine-Tuning Pipeline (CIFAR-10 Test + RSNA 8-GPU Training)

This folder contains our fine-tuning code for ResNet-50.

I already verified that the pipeline works correctly by running CIFAR-10.
Now the code is ready for the real RSNA dataset using **8 GPUs** on Perlmutter.

---

## 1. CIFAR-10 Test Result (Sanity Check)

The training loop, dataloader, optimizer, scheduler, and checkpoint saving all work.
Epoch 1: Train Acc 0.74, Val Acc 0.80
Epoch 2: Train Acc 0.86, Val Acc 0.83
Epoch 3: Train Acc 0.92, Val Acc 0.91


This confirms that the pipeline is correct.

---

## 2. How to Run on Real RSNA Dataset (with 8 GPUs)

Once the RSNA CSV + PNG files are ready, run the following on Perlmutter:

### Step 1 — Allocate 8 GPUs

### Step 2 — Run the RSNA fine-tuning script
python train_resnet50_finetune.py
--dataset csv
--train_csv RSNA_train.csv
--val_csv RSNA_val.csv
--img_root /path/to/rsna_images
--num_classes 2
--batch_size 64
--epochs 10
（This is only an example.）
The script automatically uses all available GPUs via `nn.DataParallel`.

**No extra flags are required to enable multi-GPU training.**



