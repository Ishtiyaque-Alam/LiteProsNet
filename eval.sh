python test.py \
  --metadata_csv /kaggle/input/datasets/manuelcldg/radiomics-nsclc/phase1_metadata.csv \
  --clinical_csv /kaggle/input/datasets/saibhossain/clinical-data-of-nsclc-lungi1/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv \
  --ckpt ./best.pth \
  --gtv_margin 10
