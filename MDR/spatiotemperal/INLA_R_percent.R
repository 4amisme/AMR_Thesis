# ==========================================
# สเต็ป 1: Model Validation (ด้วยเทคนิค Logit-Gaussian)
# ==========================================
library(INLA)
library(dplyr)

cat("กำลังโหลดข้อมูล...\n")
df_val <- read.csv("aba_all_region_spatiotemporal_prepared.csv")

# ----------------------------------------------------
# 1. แปลงข้อมูลด้วยเทคนิค Logit Transformation
# ----------------------------------------------------
df_val <- df_val %>%
  mutate(
    # 1. ปรับสัดส่วน 0-1 (และตัดขอบกัน Error ด้วย 0.0001 ถึง 0.9999)
    R_scaled = percentage / 100,
    R_scaled = pmin(pmax(R_scaled, 0.0001), 0.9999),
    
    # 2. แปลงเป็น Logit (ถ่างค่าให้กลายเป็น Gaussian)
    R_logit = log(R_scaled / (1 - R_scaled)),
    
    # 3. แบ่ง Train/Test (สมมติซ่อนปี 2023-2024 เป็น Test)
    set = ifelse(year %in% 2023:2024, "test", "train"),
    
    # แอบจดข้อสอบไว้ก่อน
    actual_logit = R_logit,
    actual_percent = percentage,
    actual_count = pattern_count,
    
    # ลบคำตอบ (R_logit) ในช่วงที่เป็น Test ให้กลายเป็น NA เพื่อให้ INLA ทาย
    R_logit = ifelse(set == "test", NA, R_logit)
  )

# ----------------------------------------------------
# 2. รัน R-INLA ด้วย Gaussian Family
# ----------------------------------------------------
cat("กำลังรัน INLA ด้วย Logit-Gaussian (รอสักครู่นะครับ)...\n")
# ใช้ bym2 สำหรับแผนที่ และ rw1 สำหรับเวลา (เพื่อความยืดหยุ่น)
formula <- R_logit ~ 1 + sin_month + cos_month + 
  f(region_id, model = "bym2", graph = "map.graph", scale.model = TRUE) + 
  f(time_id, model = "rw1") + 
  f(mdr_id_numeric, model = "iid")

model_val <- inla(formula, family = "gaussian", data = df_val,
                  control.predictor = list(compute = TRUE))

# ----------------------------------------------------
# 3. แปลงผลทำนายกลับเป็นเปอร์เซ็นต์ (Inverse Logit)
# ----------------------------------------------------
# ดึงค่าทำนาย (เป็นสเกล Logit)
df_val$predicted_logit <- model_val$summary.fitted.values$mean

# แปลงกลับเป็นสเกล 0-1
df_val$predicted_scaled <- exp(df_val$predicted_logit) / (1 + exp(df_val$predicted_logit))

# แปลงเป็นเปอร์เซ็นต์ 0-100%
df_val$predicted_percent <- df_val$predicted_scaled * 100

# ----------------------------------------------------
# 4. ประเมินผลสอบ (WAPE)
# ----------------------------------------------------
df_test <- df_val %>% filter(set == "test")

# คำนวณ Error จำนวนเคสจริง vs เคสทำนาย
df_test$predicted_cases <- (df_test$predicted_percent / 100) * df_test$total_rows_in_region_month
df_test$abs_error_cases <- abs(df_test$actual_count - df_test$predicted_cases)

# หาค่า WAPE
total_error <- sum(df_test$abs_error_cases, na.rm = TRUE)
total_actual <- sum(df_test$actual_count, na.rm = TRUE)
wape_overall <- (total_error / total_actual) * 100

cat("\n===================================\n")
cat("    🎯 ผลสอบของโมเดล (Logit-Gaussian) 🎯\n")
cat("===================================\n")
cat(sprintf("📊 WAPE: %.2f%%\n", wape_overall))

if(wape_overall <= 15) {
  cat("\n✅ สรุป: ยอดเยี่ยม! โมเดลผ่านเกณฑ์ < 15% แล้ว!\n")
} else {
  cat("\n⚠️ สรุป: WAPE ลดลงมาแล้ว แต่อาจจะต้องจูนเพิ่มอีกนิด\n")
}