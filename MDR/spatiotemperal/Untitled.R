# ==========================================
# 1. โหลด Library และข้อมูล
# ==========================================
library(INLA)
library(dplyr)

cat("กำลังโหลดข้อมูล...\n")
df_final <- read.csv("aba_all_region_spatiotemporal_prepared.csv")

# ==========================================
# 2. สร้างข้อมูลอนาคต 5 ปี (เพิ่มตัวแปรฤดูกาลครบ 2 แบบ)
# ==========================================
max_time_id <- max(df_final$time_id)
last_month <- 12 # สมมติเดือนล่าสุดของข้อมูลคือธันวาคม (ปรับตามจริงได้)

future_time_ids <- (max_time_id + 1):(max_time_id + 60)
regions <- 1:13
all_mdrs <- unique(df_final$mdr_id_numeric)

df_future <- expand.grid(time_id = future_time_ids, 
                         region_id = regions, 
                         mdr_id_numeric = all_mdrs)

df_future <- df_future %>%
  mutate(
    future_year = ceiling((time_id - max_time_id) / 12),
    month_numeric = ((last_month + (time_id - max_time_id) - 1) %% 12) + 1,
    
    # --- ฤดูกาลแบบที่ 1: Sine/Cosine ---
    sin_month = sin(2 * pi * month_numeric / 12),
    cos_month = cos(2 * pi * month_numeric / 12),
    
    # --- ฤดูกาลแบบที่ 2: Thai Seasons (1=ร้อน, 2=ฝน, 3=หนาว) ---
    season = case_when(
      month_numeric %in% c(3, 4, 5) ~ "summer",
      month_numeric %in% c(6, 7, 8, 9) ~ "rainy",
      month_numeric %in% c(10, 11, 12, 1, 2) ~ "winter"
    ),
    season_id = case_when(
      season == "summer" ~ 1,
      season == "rainy" ~ 2,
      season == "winter" ~ 3
    ),
    
    # ให้ค่าที่ต้องการทำนายเป็น NA (เพื่อบังคับให้ INLA พยากรณ์)
    y_beta = NA  
  )

# นำข้อมูลอดีตและอนาคตมาต่อกัน
df_combined <- bind_rows(df_final, df_future)

# ==========================================
# 3. รันโมเดลแข่งขันกัน 2 รูปแบบ (Model Comparison)
# ==========================================

# โมเดลที่ 1: ใช้ Sine / Cosine
cat("กำลังรันโมเดลที่ 1 (Sine/Cosine)... อาจใช้เวลาสักครู่\n")
formula_sincos <- y_beta ~ 1 + sin_month + cos_month + 
  f(region_id, model = "besag", graph = "map.graph", scale.model = TRUE) + 
  f(time_id, model = "ar1") + 
  f(mdr_id_numeric, model = "iid")

model_sincos <- inla(formula_sincos, family = "beta", data = df_combined,
                     control.predictor = list(link = 1, compute = TRUE),
                     control.compute = list(dic = TRUE, waic = TRUE))

# โมเดลที่ 2: ใช้ Season_id แบบขั้นบันได (Thai Seasons)
cat("กำลังรันโมเดลที่ 2 (Thai Seasons)... อาจใช้เวลาสักครู่\n")
formula_season <- y_beta ~ 1 + as.factor(season_id) + 
  f(region_id, model = "besag", graph = "map.graph", scale.model = TRUE) + 
  f(time_id, model = "ar1") + 
  f(mdr_id_numeric, model = "iid")

model_season <- inla(formula_season, family = "beta", data = df_combined,
                     control.predictor = list(link = 1, compute = TRUE),
                     control.compute = list(dic = TRUE, waic = TRUE))

# ==========================================
# 4. สรุปผลและเลือกโมเดลที่ดีที่สุด
# ==========================================
cat("\n===================================\n")
cat("       🏆 ผลการเปรียบเทียบโมเดล 🏆\n")
cat("===================================\n")
cat("DIC โมเดล Sine/Cosine  :", model_sincos$dic$dic, "\n")
cat("DIC โมเดล 3 ฤดู (ไทย)  :", model_season$dic$dic, "\n")

# เลือกโมเดลที่ DIC ต่ำกว่า
if(model_sincos$dic$dic < model_season$dic$dic) {
  cat("\n✨ ผลลัพธ์: โมเดล Sine/Cosine ชนะ! (ฟิตกับข้อมูลมากกว่า)\n")
  best_model <- model_sincos
} else {
  cat("\n✨ ผลลัพธ์: โมเดล 3 ฤดูแบบไทย ชนะ! (ฟิตกับข้อมูลมากกว่า)\n")
  best_model <- model_season
}

# ==========================================
# 5. สกัดค่าพยากรณ์และบันทึกเป็น CSV
# ==========================================
cat("\nกำลังสกัดค่าพยากรณ์จากโมเดลที่ชนะและบันทึกไฟล์...\n")

# ฟังก์ชันแก้ค่าติดลบ (Inverse Logit)
inv_logit <- function(x) { exp(x) / (1 + exp(x)) }

# ดึงค่าพยากรณ์กลับเข้าไปใน dataframe
df_combined$fitted_mean <- best_model$summary.fitted.values$mean

# แปลงสัดส่วนกลับเป็นเปอร์เซ็นต์ (พร้อมจัดการค่าที่หลุดกรอบ 0-1)
df_combined$predicted_percent <- ifelse(df_combined$fitted_mean < 0 | df_combined$fitted_mean > 1, 
                                        inv_logit(df_combined$fitted_mean), 
                                        df_combined$fitted_mean) * 100

# บันทึกเป็น Master File ไว้พล็อตใน Python
write.csv(df_combined, "all_historical_and_predicted_results.csv", row.names = FALSE)

cat("✅ สำเร็จ! บันทึกไฟล์ 'all_historical_and_predicted_results.csv' เรียบร้อย นำไปพล็อตต่อใน Python ได้เลยครับ\n")