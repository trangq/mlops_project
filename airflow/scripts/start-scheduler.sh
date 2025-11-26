#!/bin/sh
# Chờ database và minio tạm thời để đảm bảo chúng đã sẵn sàng
sleep 10

# Khởi động scheduler
exec airflow scheduler
