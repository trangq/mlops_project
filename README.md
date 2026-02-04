# MLOps Project
# 🚢 Titanic MLOps Project - Dashboard Access

Dưới đây là danh sách các dịch vụ và thông tin đăng nhập để quản lý hệ thống:

| Dịch vụ | Đường dẫn (URL) | Tài khoản | Mật khẩu |
| :--- | :--- | :--- | :--- |
| **API Documentation** | [http://localhost:8000/docs](http://localhost:8000/docs) | Không có | Không có |
| **Airflow Workflow** | [http://localhost:8080](http://localhost:8080) | `airflow` | `airflow` |
| **MLflow Tracking** | [http://localhost:5001](http://localhost:5001) | Không có | Không có |
| **MinIO Console** | [http://localhost:9001](http://localhost:9001) | `minioadmin` | `minioadmin` |

---

### 🔐 Thông tin GitHub Actions (Secrets)
Để hệ thống CI/CD gửi mail thông báo khi Test Pass, bạn cần cài đặt 2 Secret trong GitHub:
1. `MAIL_USERNAME`: Email của bạn.
2. `MAIL_PASSWORD`: App Password (mã 16 ký tự từ Google).

### 🛠 Lệnh kiểm tra nhanh
- **Chạy toàn bộ:** `docker compose up -d`
- **Xem log API:** `docker logs -f fastapi-service`
- **Chạy test tại máy:** `set PYTHONPATH=. && pytest tests/ -v`