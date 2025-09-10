from apscheduler.schedulers.background import BackgroundScheduler
from pipeline import update_faiss_index

scheduler = BackgroundScheduler()
# Update FAISS every day at 8 AM
scheduler.add_job(update_faiss_index, 'cron', hour=8)
scheduler.start()

print("Scheduler started, FAISS will update daily at 8 AM")
# Keep the scheduler running
import time
while True:
    time.sleep(60)
