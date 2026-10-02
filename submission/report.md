# Báo cáo Lab 16 — AWS CPU + LightGBM

1. Tôi dùng AWS, region us-east-1 (us-east-1a), compute node `m7i-flex.large` (2 vCPU / 8 GB; thay cho `t3.medium` vì tài khoản Free plan chỉ cho chạy instance free-tier eligible), bastion `t3.micro`; source commit `55539f6` + `benchmark.py`.
2. Dataset Credit Card Fraud có 284,807 dòng, 30 feature, tỷ lệ fraud 0.173%; chia stratified train/validation/test 64/16/20 (182,276 / 45,569 / 56,962 dòng), seed 42.
3. Load dữ liệu mất 0.939 giây; training mất 1.530 giây (`LGBMClassifier`, early stopping 50 vòng theo AUC trên validation); best iteration là 1.
4. Trên tập test: AUC 0.9391, Accuracy 0.9992, F1 0.7692, Precision 0.7732, Recall 0.7653 (ngưỡng 0.5). Accuracy cao chủ yếu do dữ liệu mất cân bằng; best iteration = 1 cho thấy model gần như chỉ dùng 1 cây, cần xử lý imbalance/tuning để cải thiện.
5. Latency 1 dòng 0.529 ms (median của 1,000 lần predict, có warm-up); batch 1,000 dòng mất 0.701 ms ≈ 1.43 triệu dòng/giây (đo 1 lần). Batch nhanh hơn nhiều so với gọi từng dòng vì overhead mỗi lần gọi chiếm phần lớn latency.
6. CPU/RAM/Network tôi quan sát lúc [sau khi chạy benchmark] là [CPU …%, RAM dùng … / 7.x GB]; ảnh đính kèm `screenshots/resources.png`.
7. Billing tại [giờ chụp] ghi nhận [… / chưa cập nhật]; ước tính khoảng $0.10–0.15/giờ cho NAT Gateway + ALB + 2 EC2; ảnh `screenshots/billing.png`.
8. Tôi đã tải kết quả và chạy `terraform destroy` lúc [giờ]; `terraform state list` trống; bằng chứng `screenshots/cleanup.png`.
