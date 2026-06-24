# ==============================================================================
# ULTIMATE FORECASTING MODEL (BINOMIAL) WITH 5-YEAR SPATIAL MAPS
# ==============================================================================
library(sf)
library(dplyr)
library(spdep)
library(INLA)
library(ggplot2)
library(readr)
library(stringr)

# ==============================================================================
# 1. MAP PREPARATION 
# ==============================================================================
cat("Loading and preparing spatial map...\n")
TH_json <- st_read("Regions_no_province_boundaries.json", quiet = TRUE) %>% 
  st_transform(., crs = st_crs(24047)) 

health_regions_sf <- TH_json %>%
  rename(region_id = HealthRegion) %>% 
  group_by(region_id) %>%
  summarize(geometry = st_union(geometry))

thailand_nb <- poly2nb(health_regions_sf, queen = TRUE)
nb2INLA("thailand_regions.graph", thailand_nb)
cat("Map prepared successfully!\n\n")

# ==============================================================================
# 2. DEFINE TARGET PATTERNS FOR EACH PATHOGEN
# ==============================================================================
target_patterns_list <- list(
  "aba" = toupper(c(
    "Aminoglycosides, Carbapenems, Cephems, Fluoroquinolones, Folate pathway antagonists, β-lactam combination agents",
    "Aminoglycosides, Carbapenems, Cephems, Fluoroquinolones, β-lactam combination agents",
    "Carbapenems, Cephems, fluoroquinolones, Folate pathway antagonists, β-lactam combination agents",
    "Carbapenems, Cephems, Fluoroquinolones, β-lactam combination agents",
    "Carbapenems, Cephems, Folate pathway antagonists, β-lactam combination agents"
  )),
  "eco" = toupper(c(
    "Cephems, Fluoroquinolones, Folate Pathway Antagonists, Penicillins",
    "Aminoglycosides, Cephems, Fluoroquinolones, Folate Pathway Antagonists, Penicillins",
    "Fluoroquinolones, Folate Pathway Antagonists, Penicillins",
    "Aminoglycosides, Cephems, Fluoroquinolones, Folate Pathway Antagonists, Penicillins, β-Lactam Combination Agents",
    "Aminoglycosides, Fluoroquinolones, Folate Pathway Antagonists, Penicillins"
  )),
  "kpn" = toupper(c(
    "Aminoglycosides, Carbapenems, Cephems, Fluoroquinolones, Folate Pathway Antagonists, Penicillins, β-Lactam Combination Agents",
    "Cephems, Fluoroquinolones, Folate Pathway Antagonists, Penicillins, β-Lactam Combination Agents",
    "Aminoglycosides, Cephems, Fluoroquinolones, Folate Pathway Antagonists, Penicillins, β-Lactam Combination Agents",
    "Carbapenems, Cephems, Fluoroquinolones, Folate Pathway Antagonists, Penicillins, β-Lactam Combination Agents",
    "Cephems, Fluoroquinolones, Folate Pathway Antagonists, Penicillins"
  )),
  "pae" = toupper(c(
    "Aminoglycosides, Carbapenems, Cephems, Fluoroquinolones, β-Lactam Combination Agents",
    "Carbapenems, Cephems, Fluoroquinolones, β-Lactam Combination Agents",
    "Aminoglycosides, Carbapenems, Cephems, Fluoroquinolones",
    "Carbapenems, Cephems, β-Lactam Combination Agents",
    "Carbapenems, Fluoroquinolones, β-Lactam Combination Agents"
  )),
  "sau" = toupper(c(
    "Lincosamides, Macrolides, Penicillins, Tetracyclines",
    "Fluoroquinolones, Lincosamides, Macrolides, Penicillins",
    "Lincosamides, Macrolides, Tetracyclines",
    "Aminoglycosides, Fluoroquinolones, Lincosamides, Macrolides, Penicillins, Tetracyclines",
    "Aminoglycosides, Fluoroquinolones, Lincosamides, Macrolides, Penicillins"
  ))
)

# ==============================================================================
# 3. MAIN FORECASTING LOOP
# ==============================================================================
for (pathogen in names(target_patterns_list)) {
  
  cat(sprintf("========== เริ่มพยากรณ์เชื้อ: %s ==========\n", toupper(pathogen)))

    file_name <- sprintf("%s_all_region_spatiotemporal_prepared.csv", pathogen)
  if(!file.exists(file_name)) {
    cat(sprintf("⚠️ ไม่พบไฟล์ %s ข้ามไปวิเคราะห์เชื้อถัดไป\n", file_name))
    next
  }
  
  df <- read_csv(file_name, show_col_types = FALSE)
  df$Resistant_Drug_Classes <- toupper(str_trim(as.character(df$Resistant_Drug_Classes)))
  
  target_drugs <- target_patterns_list[[pathogen]]
  df_filtered <- df %>% filter(Resistant_Drug_Classes %in% target_drugs)
  
  if(nrow(df_filtered) == 0) {
    cat("⚠️ ไม่พบข้อมูล 5 Pattern ข้ามเชื้อนี้\n")
    next
  }
  
  df_filtered$drug_group <- match(df_filtered$Resistant_Drug_Classes, target_drugs)
  n_groups <- 5
  
  n_forecast <- 60
  last_time <- max(df_filtered$time_id, na.rm = TRUE)
  future_times <- (last_time + 1):(last_time + n_forecast)
  
  forecast_stratified_df <- expand.grid(
    time_id = future_times,
    region_id = 1:13,
    drug_group = 1:n_groups
  ) %>%
    mutate(
      pattern_count = NA,
      total_rows_in_region_month = round(mean(df_filtered$total_rows_in_region_month, na.rm = TRUE)),
      month_numeric = ((time_id - 1) %% 12) + 1,
      sin_month = sin(2 * pi * month_numeric / 12),
      cos_month = cos(2 * pi * month_numeric / 12),
      Resistant_Drug_Classes = target_drugs[drug_group]
    )
  
  full_data <- bind_rows(df_filtered, forecast_stratified_df)
  
  # รัน INLA
  cat(sprintf("กำลังรัน INLA พยากรณ์อนาคต 5 ปี สำหรับ %s...\n", toupper(pathogen)))
  strat_forecast_formula <- pattern_count ~ 1 + Resistant_Drug_Classes + sin_month + cos_month + 
    f(region_id, model = "bym", graph = "thailand_regions.graph") + 
    f(time_id, model = "rw2", group = drug_group, control.group = list(model = "exchangeable"))
  
  res_model <- inla(
    formula = strat_forecast_formula,
    family = "binomial",
    data = full_data,
    Ntrials = full_data$total_rows_in_region_month,
    control.predictor = list(compute = TRUE, link = 1)
  )
  
  # สกัดข้อมูลเป็น %
  export_df <- full_data %>%
    mutate(
      predicted_percent = res_model$summary.fitted.values$mean * 100,
      data_type = ifelse(time_id > last_time, "forecast", "historical")
    )
  
  write.csv(export_df, sprintf("%s_5Years_Forecast_Results.csv", pathogen), row.names = FALSE)
  
  # ==============================================================================
  # 4. วาดแผนที่ 5 ปี สไตล์อาจารย์ (แสดงเปอร์เซ็นต์ที่เปลี่ยนไป +/-)
  # ==============================================================================
  cat("กำลังสร้างแผนที่พยากรณ์ 5 ปี พร้อมตัวเลข % Change...\n")
  
  for (drug in target_drugs) {
    
    # 1. ระบุเดือนที่เป็นเป้าหมาย (เดือนที่ 12, 24, 36, 48, 60 ของอนาคต)
    target_months <- c(last_time + 12, last_time + 24, last_time + 36, last_time + 48, last_time + 60)
    
    # 2. ดึงข้อมูลของยาตัวนี้ออกมา
    drug_forecast_panel <- export_df %>% filter(Resistant_Drug_Classes == drug)
    
    # 3. หาค่า Baseline (ค่าการดื้อยาในเดือนสุดท้ายของข้อมูลจริง) เพื่อเอาไว้ใช้เทียบ +/-
    baseline_data <- drug_forecast_panel %>%
      filter(time_id == last_time) %>%
      select(region_id, baseline_pct = predicted_percent)
    
    # 4. กรองเฉพาะปีอนาคตและคำนวณ % Change
    df_plot <- drug_forecast_panel %>% 
      filter(time_id %in% target_months) %>%
      left_join(baseline_data, by = "region_id") %>%
      mutate(
        future_year = (time_id - last_time) / 12,
        Year_Label = paste("Future Year", future_year),
        
        # คำนวณความเปลี่ยนแปลง (จุดเปอร์เซ็นต์)
        pct_change = predicted_percent - baseline_pct,
        # จัดฟอร์แมตตัวอักษรให้มีเครื่องหมาย +/- และทศนิยม 2 ตำแหน่ง
        label_text = sprintf("%+.2f%%", pct_change),
        
        # แปลงกลับเป็น 0-1 เฉพาะตอนเอาไประบายสี เพื่อให้ใช้ scales::percent ของอาจารย์ได้สวยๆ
        prob_for_fill = predicted_percent / 100 
      )
    
    # 5. เชื่อมข้อมูลเข้ากับไฟล์แผนที่
    map_plot_data <- left_join(health_regions_sf, df_plot, by = "region_id")
    
    # 6. วาดแผนที่ (ตามแบบต้นฉบับอาจารย์)
    p_map_5y <- ggplot(data = map_plot_data) +
      geom_sf(aes(fill = prob_for_fill), color = "white", linewidth = 0.1) +
      
      # ใส่ตัวเลข +/- สีขาวตรงกลางแต่ละเขต (สไตล์อาจารย์)
      stat_sf_coordinates(aes(label = label_text), 
                          geom = "text", 
                          color = "white", 
                          fontface = "bold", 
                          size = 2.8, 
                          check_overlap = TRUE) +
      
      facet_wrap(~ Year_Label, nrow = 1) + 
      
      # ใช้โทนสี Plasma และฟอร์แมตแกน % ตามของอาจารย์
      scale_fill_viridis_c(
        option = "plasma", 
        labels = scales::percent, 
        name = "MDR Chance",
        limits = NULL 
      ) + 
      theme_void() +
      labs(
        title = sprintf("5-Year Forecast with Projected Change (%s)", toupper(pathogen)),
        subtitle = paste("Drug Pattern:", drug, "\n(*% Change is calculated relative to the last historical month)")
      ) +
      theme(
        plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
        plot.subtitle = element_text(size = 10, hjust = 0.5, margin = margin(b=15)),
        legend.position = "bottom",
        legend.key.width = unit(2, "cm"),
        strip.text = element_text(size = 12, face = "bold")
      )
    
    # 7. บันทึกรูปภาพ
    clean_name <- substr(gsub("[, ]", "_", drug), 1, 40)
    file_out <- sprintf("%s_5Y_Map_%s.png", pathogen, clean_name)
    ggsave(file_out, plot = p_map_5y, width = 16, height = 6, dpi = 300)
    
  }
  cat(sprintf("✅ เสร็จสิ้น %s - บันทึกแผนที่ 5 รูป สไตล์อาจารย์ เรียบร้อย!\n\n", toupper(pathogen)))
  
  