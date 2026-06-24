# ==========================================
# สเต็ป 1: Model Validation (แยกทำทีละ Pattern)
# ==========================================
library(INLA)
library(dplyr)

cat("กำลังโหลดข้อมูลเพื่อทำ Validation...\n")
df_master <- read.csv("aba_all_region_spatiotemporal_prepared.csv")

# 1. เตรียมข้อมูล Logit Transformation ให้พร้อมทั้งหมดก่อน
df_master <- df_master %>%
  mutate(
    # สร้าง region_year ไว้เผื่อใช้
    region_year = as.numeric(as.factor(paste(region_id, year, sep="_"))),
    
    # ปรับสัดส่วน 0-1 (ตัดขอบกัน Error ให้อยู่ที่ 0.001 - 0.999)
    R_scaled = percentage / 100,
    R_scaled = pmin(pmax(R_scaled, 0.001), 0.999),
    
    # แปลงเป็น Logit ให้กลายเป็น Gaussian
    R_logit = log(R_scaled / (1 - R_scaled)),
    
    # แบ่ง Train/Test (สมมติซ่อนปี 2023-2024 เป็น Test)
    set = ifelse(year %in% 2023:2024, "test", "train"),
    
    # แอบจดข้อสอบไว้ตรวจ
    actual_count = pattern_count,
    actual_percent = percentage,
    
    # ลบคำตอบ (R_logit) ในช่วง Test ให้เป็น NA (เพื่อให้ INLA ทาย)
    R_logit = ifelse(set == "test", NA, R_logit)
  )

# รายชื่อ Pattern ทั้งหมด (สมมติว่ามี 5 แบบ)
all_patterns <- sort(unique(df_master$mdr_id_numeric))

# สร้าง Data Frame เปล่าๆ ไว้เก็บผลลัพธ์
results_wape <- data.frame(Pattern_ID = integer(), WAPE = numeric())

# 2. เริ่มลูป! แยกทำทีละ Pattern
for (p_id in all_patterns) {
  
  cat(sprintf("\n\n>>> กำลังวิเคราะห์ Pattern ที่ %d <<<\n", p_id))
  
  # ดึงข้อมูลมาเฉพาะ Pattern นี้
  df_sub <- df_master %>% filter(mdr_id_numeric == p_id)
  
  # ----------------------------------------------------
  # รัน INLA (สมการจะเบาลงมาก เพราะไม่ต้องมี f(mdr_id_numeric) แล้ว)
  # ----------------------------------------------------
  # ใช้ BYM2 สำหรับพื้นที่ และ RW1 สำหรับเวลา
  formula_sub <- R_logit ~ 1 + sin_month + cos_month + 
    f(region_id, model = "bym2", graph = "map.graph", scale.model = TRUE) + 
    f(time_id, model = "rw1", scale.model = TRUE) 
  # (ลองเติม f(region_year, model="iid") ได้ถ้าข้อมูลใน Pattern นี้มีเยอะพอ)
  
  # รันโมเดล (ปรับ strategy="gaussian" กันเหนียวไว้ก่อน)
  model_sub <- inla(formula_sub, 
                    family = "gaussian", 
                    data = df_sub,
                    control.predictor = list(compute = TRUE),
                    control.compute = list(dic = TRUE, waic = TRUE),
                    control.inla = list(strategy = "gaussian", int.strategy = "eb"))
  
  # ----------------------------------------------------
  # ประมวลผลและคำนวณ WAPE ของ Pattern นี้
  # ----------------------------------------------------
  df_sub$predicted_logit <- model_sub$summary.fitted.values$mean
  df_sub$predicted_scaled <- exp(df_sub$predicted_logit) / (1 + exp(df_sub$predicted_logit))
  df_sub$predicted_percent <- df_sub$predicted_scaled * 100
  
  # ตัดมาตรวจข้อสอบเฉพาะ Test Set
  df_test_sub <- df_sub %>% filter(set == "test")
  
  df_test_sub$predicted_cases <- (df_test_sub$predicted_percent / 100) * df_test_sub$total_rows_in_region_month
  df_test_sub$abs_error_cases <- abs(df_test_sub$actual_count - df_test_sub$predicted_cases)
  
  total_error_sub <- sum(df_test_sub$abs_error_cases, na.rm = TRUE)
  total_actual_sub <- sum(df_test_sub$actual_count, na.rm = TRUE)
  
  # ถ้ามีข้อมูลจริงให้คำนวณ ก็หา WAPE
  if(total_actual_sub > 0){
    wape_sub <- (total_error_sub / total_actual_sub) * 100
  } else {
    wape_sub <- NA
  }
  
  cat(sprintf("✅ เสร็จสิ้น Pattern %d! WAPE = %.2f%%\n", p_id, wape_sub))
  
  # เก็บผลลัพธ์ใส่ตารางรวม
  results_wape <- rbind(results_wape, data.frame(Pattern_ID = p_id, WAPE = wape_sub))
}

# สรุปผลสอบ WAPE ราย Pattern
cat("\n===================================\n")
cat("  🎯 สรุปผลสอบ WAPE แยกตาม Pattern 🎯\n")
cat("===================================\n")
print(results_wape)

# หา Overall WAPE (เฉลี่ยจากทุก Pattern แบบคร่าวๆ)
cat(sprintf("\n🌟 ค่า WAPE เฉลี่ยรวม: %.2f%%\n", mean(results_wape$WAPE, na.rm = TRUE)))



# ==========================================
# สเต็ป 2: Future Forecasting (แยกทำทีละ Pattern)
# ==========================================

cat("กำลังเตรียมข้อมูลเพื่อทำนายอนาคต (2025-2029)...\n")

# 1. โหลดข้อมูลจริง (Historical Data)
df_historical <- read.csv("aba_all_region_spatiotemporal_prepared.csv")

# 2. สร้างข้อมูลอนาคต 5 ปี (60 เดือน) แบบตาราง Grid
max_time_id <- max(df_historical$time_id)
last_month <- 12 # สมมติเดือนล่าสุดของข้อมูลคือธันวาคม 2024
future_time_ids <- (max_time_id + 1):(max_time_id + 60)
regions <- 1:13
all_patterns <- sort(unique(df_historical$mdr_id_numeric))

df_future <- expand.grid(time_id = future_time_ids, 
                         region_id = regions, 
                         mdr_id_numeric = all_patterns)

# เติมตัวแปรต่างๆ ให้ข้อมูลอนาคต
df_future <- df_future %>%
  mutate(
    future_year = ceiling((time_id - max_time_id) / 12),
    year = 2024 + future_year, # สมมติปีสุดท้ายคือ 2024 
    month = ((last_month + (time_id - max_time_id) - 1) %% 12) + 1,
    
    # คำนวณตัวแปรฤดูกาล
    sin_month = sin(2 * pi * month / 12),
    cos_month = cos(2 * pi * month / 12),
    
    # ให้ค่าเปอร์เซ็นต์ดื้อยาเป็น NA เพื่อบังคับให้ INLA พยากรณ์
    percentage = NA,
    pattern_count = NA,
    total_rows_in_region_month = NA,
    
    # ระบุว่าเป็นชุดข้อมูลอนาคต
    data_type = "forecast" 
  )

# ระบุว่าข้อมูลเดิมเป็นข้อมูลอดีต
df_historical$data_type <- "historical"
df_historical$future_year <- NA

# รวมร่างอดีตและอนาคตเข้าด้วยกัน
df_master_forecast <- bind_rows(df_historical, df_future)

# 3. เตรียมข้อมูล Logit Transformation (สำหรับอดีตและอนาคต)
df_master_forecast <- df_master_forecast %>%
  mutate(
    R_scaled = percentage / 100,
    R_scaled = pmin(pmax(R_scaled, 0.001), 0.999), # ตัดขอบ
    R_logit = log(R_scaled / (1 - R_scaled))       # แปลง Logit (อนาคตจะเป็น NA อัตโนมัติ)
  )

# สร้าง Data Frame เปล่าเก็บผลลัพธ์สุดท้าย
df_final_results <- data.frame()

# ----------------------------------------------------
# 4. เริ่มลูปพยากรณ์! (แยกทำทีละ Pattern)
# ----------------------------------------------------
for (p_id in all_patterns) {
  
  cat(sprintf("\n\n>>> 🚀 กำลังพยากรณ์อนาคต Pattern ที่ %d <<<\n", p_id))
  
  # ดึงข้อมูลเฉพาะ Pattern นี้
  df_sub <- df_master_forecast %>% filter(mdr_id_numeric == p_id)
  
  # สร้างสมการ INLA (BYM2 + RW1)
  formula_forecast <- R_logit ~ 1 + sin_month + cos_month + 
    f(region_id, model = "bym2", graph = "map.graph", scale.model = TRUE) + 
    f(time_id, model = "rw1", scale.model = TRUE)
  
  # รันโมเดล!
  model_forecast <- inla(formula_forecast, 
                         family = "gaussian", 
                         data = df_sub,
                         control.predictor = list(compute = TRUE),
                         control.inla = list(strategy = "gaussian", int.strategy = "eb"))
  
  # สกัดผลพยากรณ์กลับมาเป็นเปอร์เซ็นต์
  df_sub$predicted_logit <- model_forecast$summary.fitted.values$mean
  df_sub$predicted_scaled <- exp(df_sub$predicted_logit) / (1 + exp(df_sub$predicted_logit))
  df_sub$predicted_percent <- df_sub$predicted_scaled * 100
  
  # ดึงช่วงความเชื่อมั่น 95% (95% Credible Interval) มาด้วย เผื่อทำกราฟ Error bar
  df_sub$predicted_lower_CI <- (exp(model_forecast$summary.fitted.values$`0.025quant`) / (1 + exp(model_forecast$summary.fitted.values$`0.025quant`))) * 100
  df_sub$predicted_upper_CI <- (exp(model_forecast$summary.fitted.values$`0.975quant`) / (1 + exp(model_forecast$summary.fitted.values$`0.975quant`))) * 100
  
  # เอาไปต่อท้ายในตารางผลลัพธ์รวม
  df_final_results <- bind_rows(df_final_results, df_sub)
  
  cat(sprintf("✅ พยากรณ์ Pattern %d เสร็จสมบูรณ์!\n", p_id))
}

# ----------------------------------------------------
# 5. บันทึกไฟล์ผลลัพธ์สุดท้าย
# ----------------------------------------------------
cat("\n===================================\n")
cat("กำลังบันทึกไฟล์ผลลัพธ์พยากรณ์ทั้งหมด...\n")

# บันทึกเป็น CSV เพื่อเอาไปวาดกราฟ/ทำแผนที่ต่อใน Python
write.csv(df_final_results, "Forecast_5Years_All_Patterns_LogitGaussian.csv", row.names = FALSE)

cat("✅ สำเร็จ! ไฟล์ 'Forecast_5Years_All_Patterns_LogitGaussian.csv' พร้อมใช้งานครับ!\n")