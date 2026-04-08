python main.py \
  --metadata_csv /kaggle/input/datasets/manuelcldg/radiomics-nsclc/phase1_metadata.csv \
  --clinical_csv /kaggle/input/datasets/saibhossain/clinical-data-of-nsclc-lungi1/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv \
  --ckpt_path ./ \
  --alpha 0.5 \
  --beta 0.5 \
  --batch_size 64 \
  --epochs 800 \
  --lr 0.0005 \
  --gtv_margin 10
