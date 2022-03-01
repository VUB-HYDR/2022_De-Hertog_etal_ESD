#!/bin/bash

# THIS SCRIPT DOES THE PREPROCESSING AND EXECUTES THE PYTHON SCRIPT TO
# SEPARATE LOCAL, NON-LOCAL AND TOTAL IMPACTS OF LCLM CHANGE FOR
# LAMACLIMA WP1 SIMULATIONS: CTL, CROP, IRR, FRST, HARV

# BEFORE EXECUTING THE SCRIPT DEFINE THE FOLLOWING VARIABLES BELOW:
# 1. METHOD
# 2. ESM MODEL
# 3. CMOR TABLE ID (--> TEMPORAL RESOLUTION)
# 4. CMORvar
# 5. SCENARIO COMBINATION

# THE SCRIPT WILL SAVE THE OUTPUT TO:
# /scratch/b/b380948/signal_separation/<SCENARIO COMBINATION>/<CMOR TABLE ID>/<CMORvar>/*<ESM MODEL>.nc

# EXECUTE THIS SCRIPT VIA: sbatch signal_seperation.sh

#SBATCH --job-name=signal_separation      # Specify job name
#SBATCH --partition=shared                # Specify partition name
#SBATCH --ntasks=2                        # Specify max. number of tasks to be invoked
#SBATCH --cpus-per-task=16                # Specify number of CPUs per task
#SBATCH --time=04:00:00                   # Set a limit on the total run time
#SBATCH --account=bm1147                  # Charge resources on this project account
#SBATCH --output=signal_separation_TSA_irr-ctl_3hr_absolute_values_scen.out    # File name for standard output
#SBATCH --error=signal_separation_TSA_irr-ctl_3hr_absolute_values_scen.out     # File name for standard error output

# Bind your OpenMP threads --> I have to admit I have no idea if that's needed or what is happening here....
export OMP_NUM_THREADS=8
export KMP_AFFINITY=verbose,granularity=core,compact,1
export KMP_STACKSIZE=64m

module unload python
module load python/.3.5.2

METHOD="multi-year_mean_signal" #, "absolute_values_ctl", "absolute_values_scen", "multi-year_mean_signal", "each_timestep_signal"
MODEL="cesm" #, "cesm" "ecearth" "mpiesem"
CMOR_TABLE="Amon" # "Lmon" "LImon" "Emon" "Eyr" "Lmon"; 
#CMOR_VAR_LIST="cLitter cVeg cProduct" # Lmon
#CMOR_VAR_LIST="cSoil cLand" # Emon
#CMOR_VAR_LIST="FSH EFLX_LH_TOT HEAT_FROM_AC TG TSA FIRE "
# CMOR_VAR_LIST="gpp npp nbp ra rh"
#CMOR_VAR_LIST="ts_ebal_x ts_ebal_n ts_ebal_meandaymax ts_ebal_meandaymin DTR"
# CMOR_VAR_LIST="TSA TG EFLX_LH_TOT FSH PRECC PRECL WIND Q2M" # cesm model output data used with cesm table
#CMOR_VAR_LIST="wbgtid_iso_400W wbgtod_iso_400W"
CMOR_VAR_LIST="wbgtid wbgtod"
#CMOR_VAR_LIST="TREFHT QREFHT TS SHFLX LHFLX"
#CMOR_VAR_LIST="tas ts prc hfss hfls hurs pr huss"
#CMOR_VAR_LIST="TS TREFHT SHFLX LHFLX QREFHT PRECC PRECL"
#CMOR_VAR_LIST="FSDS FSR WBT Q2M RH2M TG"
CMOR_VAR_LIST="TSMN TSMX TREFHTMX TREFHTMN TSMNAVG TSMXAVG"
#CMOR_VAR_LIST="TSA TG TS_ebal FSDS FSR FLDS FIRE FSH EFLX_LH_TOT FGR FSM WASTEHEAT HEAT_FROM_AC" 
#CMOR_VAR_LIST="ts_ebal SWout LWout albedo" # tasmax hurs" 
#CMOR_VAR_LIST="sshf slhf ssrd ssr strd str"  
#CMOR_VAR_LIST="ts_ebal rlds rlus rsds rsus tas ts hfss hfls albedo"

SIM1="frst" # "frst" "crop" "irr" "harv"
SIM2="ctl" # "ctl" "crop" "frst"
SCENARIO_COMBNATION="${SIM1}-${SIM2}" #, "irr-crop" "irr-ctl" "frst-ctl" "harv-frst" "harv-ctl"
agg="none"  # default: none, avg, seas ; currently only for monthly data

# CDO COMMAND FOR MULTI-YEAR MEAN
if [ "${CMOR_TABLE}" == "Amon" ] || [ "${CMOR_TABLE}" == "Emon" ] || [ "${CMOR_TABLE}" == "Lmon" ] || [ "${CMOR_TABLE}" == "LImon" ] || [ "${CMOR_TABLE}" == "Omon" ]; then
  if [ "${agg}" == "none" ]; then
    CDO_COMMAND_1="ymonmean"
    CDO_COMMAND_2_ADD="ymonadd"
    CDO_COMMAND_2_SUB="ymonsub"
  
    # PARAMS FOR cdo splitsel TO DIVIDE DATASET INTO 30 YEAR SLICES...
    # AND REMOVE FIRST 10 YEARS OF "SPIN-UP"
    NSETS=$((12*30))
    NOFFSET=$((12*10))
  elif [ "${agg}" == 'avg' ]; then
    CDO_COMMAND_1="timmean"
    CDO_COMMAND_2_ADD="add"
    CDO_COMMAND_2_SUB="sub"
    echo $agg
    # PARAMS FOR cdo splitsel TO DIVIDE DATASET INTO 30 YEAR SLICES...
    # AND REMOVE FIRST 10 YEARS OF "SPIN-UP"
    NSETS=$((12*30))
    NOFFSET=$((12*10))
  elif [ "${agg}" == 'seas' ]; then
    CDO_COMMAND_1="seasmean"
    CDO_COMMAND_2_ADD="add"
    CDO_COMMAND_2_SUB="sub"
    echo $agg
    # PARAMS FOR cdo splitsel TO DIVIDE DATASET INTO 30 YEAR SLICES...
    # AND REMOVE FIRST 10 YEARS OF "SPIN-UP"
    NSETS=$((12*30))
    NOFFSET=$((12*10))

  fi

elif [ "${CMOR_TABLE}" == "Eyr" ] || [ "${CMOR_TABLE}" == "Oyr" ] || [ "${CMOR_TABLE}" == "extremes" ] || [ "${CMOR_TABLE}" == "Eyrmin" ] || [ "${CMOR_TABLE}" == "Eyrmax" ]; then
  CDO_COMMAND_1="timmean"
  CDO_COMMAND_2_ADD="add"
  CDO_COMMAND_2_SUB="sub"
  
  # PARAMS FOR cdo splitsel TO DIVIDE DATASET INTO 30 YEAR SLICES...
  # AND REMOVE FIRST 10 YEARS OF "SPIN-UP"
  NSETS=30
  NOFFSET=10

elif [ "${CMOR_TABLE}" == "3hr" ] || [ "${CMOR_TABLE}" == "6hrLev" ] || [ "${CMOR_TABLE}" == "6hrPlev" ] || [ "${CMOR_TABLE}" == "6hrPlevPt" ] || [ "${CMOR_TABLE}" == "E3hr" ] || [ "${CMOR_TABLE}" == "E3hrPt" ]; then
  CDO_COMMAND_1="yhourmean"
  CDO_COMMAND_2_ADD="yhouradd"
  CDO_COMMAND_2_SUB="yhoursub"
  
  # PARAMS FOR cdo splitsel TO DIVIDE DATASET INTO 30 YEAR SLICES...
  # AND REMOVE FIRST TEN YEARS OF "SPIN-UP"
  if [ "${CMOR_TABLE}" == "3hr" ] || [ "${CMOR_TABLE}" == "E3hr" ] || [ "${CMOR_TABLE}" == "E3hrPt" ]; then
    NSETS=$((365*8*30))
    NOFFSET=$((365*8*10))
  
  else # 6hr
    NSETS=$((365*4*30))
    NOFFSET=$((365*4*10))

  fi
  
elif [ "${CMOR_TABLE}" == "CFday" ] || [ "${CMOR_TABLE}" == "day" ] || [ "${CMOR_TABLE}" == "Eday" ] || [ "${CMOR_TABLE}" == "Oday" ] || [ "${CMOR_TABLE}" == "dayx" ] || [ "${CMOR_TABLE}" == "dayn" ]; then
  CDO_COMMAND_1="ydaymean"
  CDO_COMMAND_2_ADD="ydayadd"
  CDO_COMMAND_2_SUB="ydaysub"
  
  NSETS=$((365*30))
  NOFFSET=$((365*10))

fi

for CMOR_VAR in ${CMOR_VAR_LIST}; do
#[ "$SCENARIO_COMBNATION" == "crop-ctl" ] && 
if [ "${MODEL}" == "mpiesm" ]; then
    DIR1=/scratch/b/b381334/signal_separation/mpi_output/${SIM1}/${CMOR_TABLE}
    DIR2=/scratch/b/b381334/signal_separation/mpi_output/${SIM2}/${CMOR_TABLE}
        
    CDO_STARTDATE=2025-01-01-00:00:00
    CDO_ENDDATE=2174-12-31-23:59:59

elif [ "${MODEL}" == "cesm" ]; then
    DIR1=/scratch/b/b381334/signal_separation/cesm_output/${SIM1}/${CMOR_TABLE}
    DIR2=/scratch/b/b381334/signal_separation/cesm_output/${SIM2}/${CMOR_TABLE}
    
    CDO_STARTDATE=0011-01-01-00:00:00
    CDO_ENDDATE=0160-12-31-23:59:59
    
elif [ "${MODEL}" == "ecearth" ]; then    
    DIR1=/scratch/b/b381334/signal_separation/ecearth_output/${SIM1}/${CMOR_TABLE}
    DIR2=/scratch/b/b381334/signal_separation/ecearth_output/${SIM2}/${CMOR_TABLE}
    
    CDO_STARTDATE=2025-01-01-00:00:00
    CDO_ENDDATE=2174-12-31-23:59:59
fi

SCRATCH="/scratch/b/b381334/signal_separation"
# CREATE TEMPORARY FOLDERS ON SCRATCH TO SAFE THE DATA
if [ ! -d "${SCRATCH}" ]; then
  echo "Folder ${SCRATCH} does not exist and will be made."
  mkdir $SCRATCH
fi
cd $SCRATCH

if [ ! -d "${SCENARIO_COMBNATION}" ]; then
  echo "Folder ${SCRATCH}/${SCENARIO_COMBNATION} does not exist and will be made."
  mkdir ${SCENARIO_COMBNATION}
else
  echo "Folder ${SCRATCH}/${SCENARIO_COMBNATION} already exists."
fi
cd ${SCENARIO_COMBNATION}

if [ ! -d "${CMOR_TABLE}" ]; then
  echo "Folder ${CMOR_TABLE} does not exist and will be made."
  mkdir ${CMOR_TABLE}
else 
  echo "Folder ${CMOR_TABLE} already exists."  
fi
cd ${CMOR_TABLE}

if [ ! -d "${CMOR_VAR}" ]; then
  echo "Folder ${CMOR_VAR} does not exist and will be made."
  mkdir ${CMOR_VAR}
else
  echo "Folder ${CMOR_VAR} already exists."
fi
cd ${CMOR_VAR}


# IF ONE OF THE FILES DO NOT ALREADY EXISTS JUST CREATE BOTH
if [ ! -f ${CMOR_VAR}_${SIM1}_${MODEL}.nc ] && [ ! -f ${CMOR_VAR}_${SIM2}_${MODEL}.nc ] && [ ! -f ${CMOR_VAR}_${SIM1}_${MODEL}_150-years.nc ] && [ ! -f ${CMOR_VAR}_${SIM2}_${MODEL}_150-years.nc ]; then

  # COPY AND MERGE ALL MODEL OUTPUT FILES
  if [ "${MODEL}" == "mpiesm" ]; then
    FILES1=`find ${DIR1}/${CMOR_VAR}/* -name "${CMOR_VAR}*"`
    FILES2=`find ${DIR2}/${CMOR_VAR}/* -name "${CMOR_VAR}*"`  
    echo $DIR1
    echo $CMOR_VAR  
    cdo mergetime ${FILES1} ${CMOR_VAR}_${SIM1}_${MODEL}.nc
    cdo mergetime ${FILES2} ${CMOR_VAR}_${SIM2}_${MODEL}.nc
    echo ${CMOR_VAR}_${SIM1}_${MODEL}.nc
    echo ${CMOR_VAR}_${SIM2}_${MODEL}.nc   

    if [ "${CMOR_TABLE}" == "Amon" ] && [ "$SCENARIO_COMBNATION" == "frst-ctl" ] ; then
      #ncks  -O -d time,10,160 ${CMOR_VAR}_${SIM2}_${MODEL}.nc ${CMOR_VAR}_${SIM2}_${MODEL}.nc   # for FRST-CTL && yearly data signal separation
      ncks  -O -d time,120,1931 ${CMOR_VAR}_${SIM2}_${MODEL}.nc ${CMOR_VAR}_${SIM2}_${MODEL}.nc  # for FRST-CTL && monthly data separation
    elif [ "${CMOR_TABLE}" == "Amon" ] && [ "$SCENARIO_COMBNATION" == "harv-frst" ] ; then
      #ncks  -O -d time,30,149 ${CMOR_VAR}_${SIM2}_${MODEL}.nc ${CMOR_VAR}_${SIM2}_${MODEL}.nc      # for  HARV-FRST && yearly data separation
      ncks  -O -d time,360,1799 ${CMOR_VAR}_${SIM2}_${MODEL}.nc ${CMOR_VAR}_${SIM2}_${MODEL}.nc     # for HARV-FRST && monthly data separation
      echo ${CMOR_VAR}_${SIM2}_${MODEL}.nc 
    elif [ "${CMOR_TABLE}" == "Lmon" ] && [ "$SCENARIO_COMBNATION" == "frst-ctl" ] ; then
      #ncks  -O -d time,10,160 ${CMOR_VAR}_${SIM2}_${MODEL}.nc ${CMOR_VAR}_${SIM2}_${MODEL}.nc   # for FRST-CTL && yearly data signal separation
      ncks  -O -d time,120,1931 ${CMOR_VAR}_${SIM2}_${MODEL}.nc ${CMOR_VAR}_${SIM2}_${MODEL}.nc  # for FRST-CTL && monthly data separation
    elif [ "${CMOR_TABLE}" == "Lmon" ] && [ "$SCENARIO_COMBNATION" == "harv-frst" ] ; then
      #ncks  -O -d time,30,149 ${CMOR_VAR}_${SIM2}_${MODEL}.nc ${CMOR_VAR}_${SIM2}_${MODEL}.nc      # for  HARV-FRST && yearly data separation
      ncks  -O -d time,360,1799 ${CMOR_VAR}_${SIM2}_${MODEL}.nc ${CMOR_VAR}_${SIM2}_${MODEL}.nc     # for HARV-FRST && monthly data separation
      echo ${CMOR_VAR}_${SIM2}_${MODEL}.nc
    elif [ "${CMOR_TABLE}" == "Amon" ] && [ "$SCENARIO_COMBNATION" == "crop-frst" ] ; then
      #ncks  -O -d time,10,160 ${CMOR_VAR}_${SIM1}_${MODEL}.nc ${CMOR_VAR}_${SIM1}_${MODEL}.nc   # for FRST-CTL && yearly data signal separation
      ncks  -O -d time,120,1931 ${CMOR_VAR}_${SIM1}_${MODEL}.nc ${CMOR_VAR}_${SIM1}_${MODEL}.nc  # for FRST-CTL && monthly data separation
    elif [ "${CMOR_TABLE}" == "6hrPlev" ] && [ "$SCENARIO_COMBNATION" == "crop-frst" ] ; then
      #ncks  -O -d time,10,160 ${CMOR_VAR}_${SIM1}_${MODEL}.nc ${CMOR_VAR}_${SIM1}_${MODEL}.nc   # for FRST-CTL && yearly data signal separation
      ncks  -O -d time,14400,230400 ${CMOR_VAR}_${SIM1}_${MODEL}.nc ${CMOR_VAR}_${SIM1}_${MODEL}.nc  # for FRST-CTL && monthly data separation
    elif [ "${CMOR_TABLE}" == "6hrPlev" ] && [ "$SCENARIO_COMBNATION" == "frst-ctl" ] ; then
      #ncks  -O -d time,10,160 ${CMOR_VAR}_${SIM1}_${MODEL}.nc ${CMOR_VAR}_${SIM1}_${MODEL}.nc   # for FRST-CTL && yearly data signal separation
      ncks  -O -d time,14400,230400 ${CMOR_VAR}_${SIM2}_${MODEL}.nc ${CMOR_VAR}_${SIM2}_${MODEL}.nc  # for FRST-CTL && monthly data separation

    fi
      
  # COPY ALL MODEL OUTPUT FILES
  elif [ "${MODEL}" == "cesm" ]; then  
    FILES1=`find ${DIR1}/* -name "${CMOR_VAR}_*"`
    FILES2=`find ${DIR2}/* -name "${CMOR_VAR}_*"`
    echo $DIR1
    echo $DIR2
    echo $FILES1
    echo $FILES2
    cp ${FILES1} ${CMOR_VAR}_${SIM1}_${MODEL}.nc
    cp ${FILES2} ${CMOR_VAR}_${SIM2}_${MODEL}.nc
    if [ "${CMOR_TABLE}" == "Lmon" ] && [ "$SCENARIO_COMBNATION" == "harv-frst" ] ; then
      #ncks  -O -d time,30,149 ${CMOR_VAR}_${SIM2}_${MODEL}.nc ${CMOR_VAR}_${SIM2}_${MODEL}.nc      # for  HARV-FRST && yearly data separation
      ncks  -O -d time,360,1799 ${CMOR_VAR}_${SIM2}_${MODEL}.nc ${CMOR_VAR}_${SIM2}_${MODEL}.nc     # for HARV-FRST && monthly data separation
      ncks  -O -d time,360,1799 ${CMOR_VAR}_${SIM1}_${MODEL}.nc ${CMOR_VAR}_${SIM1}_${MODEL}.nc 
     elif [ "${CMOR_TABLE}" == "Amon" ] && [ "$SCENARIO_COMBNATION" == "harv-frst" ] ; then
      #ncks  -O -d time,30,149 ${CMOR_VAR}_${SIM2}_${MODEL}.nc ${CMOR_VAR}_${SIM2}_${MODEL}.nc      # for  HARV-FRST && yearly data separation
      ncks  -O -d time,360,1799 ${CMOR_VAR}_${SIM2}_${MODEL}.nc ${CMOR_VAR}_${SIM2}_${MODEL}.nc     # for HARV-FRST && monthly data separation
      ncks  -O -d time,360,1799 ${CMOR_VAR}_${SIM1}_${MODEL}.nc ${CMOR_VAR}_${SIM1}_${MODEL}.nc
    fi
      # COPY ALL MODEL OUTPUT FILES
  elif [ "${MODEL}" == "ecearth" ]; then  
    FILES1=`find ${DIR1}/* -name "${CMOR_VAR}_*"`
    FILES2=`find ${DIR2}/* -name "${CMOR_VAR}_*"`
    echo $DIR1
    echo $DIR2
    echo $FILES1
    echo $FILES2
    cp ${FILES1} ${CMOR_VAR}_${SIM1}_${MODEL}.nc
    cp ${FILES2} ${CMOR_VAR}_${SIM2}_${MODEL}.nc
    if [ "${CMOR_TABLE}" == "Lmon" ] && [ "$SCENARIO_COMBNATION" == "harv-frst" ] ; then
      #ncks  -O -d time,30,149 ${CMOR_VAR}_${SIM2}_${MODEL}.nc ${CMOR_VAR}_${SIM2}_${MODEL}.nc      # for  HARV-FRST && yearly data separation
      ncks  -O -d time,360,1799 ${CMOR_VAR}_${SIM2}_${MODEL}.nc ${CMOR_VAR}_${SIM2}_${MODEL}.nc     # for HARV-FRST && monthly data separation
      ncks  -O -d time,360,1799 ${CMOR_VAR}_${SIM1}_${MODEL}.nc ${CMOR_VAR}_${SIM1}_${MODEL}.nc 
     elif [ "${CMOR_TABLE}" == "Amon" ] && [ "$SCENARIO_COMBNATION" == "harv-frst" ] ; then
      #ncks  -O -d time,30,149 ${CMOR_VAR}_${SIM2}_${MODEL}.nc ${CMOR_VAR}_${SIM2}_${MODEL}.nc      # for  HARV-FRST && yearly data separation
      ncks  -O -d time,360,1799 ${CMOR_VAR}_${SIM2}_${MODEL}.nc ${CMOR_VAR}_${SIM2}_${MODEL}.nc     # for HARV-FRST && monthly data separation
      ncks  -O -d time,360,1799 ${CMOR_VAR}_${SIM1}_${MODEL}.nc ${CMOR_VAR}_${SIM1}_${MODEL}.nc
    fi
  fi
  
  
  # CHECK NUMBER OF LEVELS (FOR JUST ONE SCENARIO) AND STOP SCRIPT WHEN MORE THAN ONE LEVEL
  NLEVELS='cdo nlevel ${CMOR_VAR}_${SIM1}_${MODEL}.nc'
  if [ ${NLEVELS} -gt 1 ]; then
    echo "Input data set has ${NLEVELS} instead of just 1 level. Please split the file into ${NLEVELS} separate files each containing one level."
    exit 1
#     for i in $(seq 1 $NLEVELS); do echo $i; done
  fi

  if [ "${METHOD}" ==  "absolute_values_ctl" ] || [ "${METHOD}" ==  "multi-year_mean_signal" ] || [ "${METHOD}" ==  "absolute_values_scen" ]; then  
  # SEPARATE VARIABLES INTO TIME 5x30 YEARS TIME SLICES AND REMOVE SPINUP
  # NAMING: ${CMOR_VAR}_${SIM1}_${MODEL}000000.nc to ${CMOR_VAR}_${SIM1}_${MODEL}000004.nc
  cdo seldate,${CDO_STARTDATE},${CDO_ENDDATE} ${CMOR_VAR}_${SIM1}_${MODEL}.nc ${CMOR_VAR}_${SIM1}_${MODEL}_150-years.nc
  cdo seldate,${CDO_STARTDATE},${CDO_ENDDATE} ${CMOR_VAR}_${SIM2}_${MODEL}.nc ${CMOR_VAR}_${SIM2}_${MODEL}_150-years.nc
  
  rm ${CMOR_VAR}_${SIM1}_${MODEL}.nc ${CMOR_VAR}_${SIM2}_${MODEL}.nc
  
#   cdo splitsel,${NSETS},${NOFFSET} ${CMOR_VAR}_${SIM1}_${MODEL}.nc ${CMOR_VAR}_${SIM1}_${MODEL}
#   cdo splitsel,${NSETS},${NOFFSET} ${CMOR_VAR}_${SIM2}_${MODEL}.nc ${CMOR_VAR}_${SIM2}_${MODEL}  
  cdo splitsel,${NSETS} ${CMOR_VAR}_${SIM1}_${MODEL}_150-years.nc ${CMOR_VAR}_${SIM1}_${MODEL}
  cdo splitsel,${NSETS} ${CMOR_VAR}_${SIM2}_${MODEL}_150-years.nc ${CMOR_VAR}_${SIM2}_${MODEL}
  fi
  
else  
  echo "Files ${CMOR_VAR}_${SIM1}_${MODEL}.nc and ${CMOR_VAR}_${SIM2}_${MODEL}.nc already exist."
  
fi  
  
# WHICH SIGNAL SEPARATION TO PERFORM?
# Equation (2)  from Winckler et al.2019 ESD: 
# delta_total = delta_local + delta_nonlocal
# with delta == SCEN - CTL, the equation can be rearranged to:
# SCEN_total - CTL = (SCEN_local - CTL) + (SCEN_nonlocal - CTL)
# multi year mean for CTL and SCEN (SCEN_local, SCEN_nonlocal, SCEN_total by interpolation)
# This equation will be rearranged and multi-year mean values applied to different variables

# METHOD="absolute_values_ctl" #, "absolute_values_ctl", "absolute_values_scen", "multi-year_mean_signal" "each_timestep_signal"

if [ "${METHOD}" ==  "each_timestep_signal" ]; then
  cdo sub ${CMOR_VAR}_${SIM1}_${MODEL}_150-years.nc ${CMOR_VAR}_${SIM2}_${MODEL}_150-years.nc ${CMOR_VAR}_${SIM1}-${SIM2}_${MODEL}.nc
  cp ${CMOR_VAR}_${SIM1}-${SIM2}_${MODEL}.nc ${CMOR_VAR}_${SIM1}-${SIM2}_${MODEL}_signal-separated.nc
  NC_FILE=${SCRATCH}/${SCENARIO_COMBNATION}/${CMOR_TABLE}/${CMOR_VAR}/${CMOR_VAR}_${SIM1}-${SIM2}_${MODEL}_signal-separated.nc
  
  echo "Calculate signal separation for variable ${CMOR_VAR} in file ${NC_FILE}"
  python /pf/b/b381334/signal_seperation/prepare_local_nonlocal_total_effects.py ${NC_FILE} ${CMOR_VAR} interpolate_local
#   python /pf/b/b380948/lamaclima/prepare_local_nonlocal_total_effects.py ${NC_FILE} ${CMOR_VAR} interpolate_total

# absolute_values_ctl: absolute values based on CTL time series
elif [ "${METHOD}" ==  "absolute_values_ctl" ] || [ "${METHOD}" ==  "multi-year_mean_signal" ]; then
  # (1) multi year mean of CTL and SCEN to calc delta_total = delta_local + delta_nonlocal
  # (2) SCEN_total = delta_total + CTL, with delta_total as multi year mean value and CTL as full time series
  # (3) SCEN_local = delta_local...
  
  for SUFFIX in {000000..000004}; do
  
    # IF THE SIGNAL SEPARATED FILE DOES NOT ALREADY EXIST, CREATE IT
    if [ ! -f ${CMOR_VAR}_${SIM1}-${SIM2}_${MODEL}_${CDO_COMMAND_1}${SUFFIX}_signal-separated.nc ]; then
    
    cdo ${CDO_COMMAND_1} ${CMOR_VAR}_${SIM1}_${MODEL}${SUFFIX}.nc ${CMOR_VAR}_${SIM1}_${MODEL}_${CDO_COMMAND_1}${SUFFIX}.nc
    cdo ${CDO_COMMAND_1} ${CMOR_VAR}_${SIM2}_${MODEL}${SUFFIX}.nc ${CMOR_VAR}_${SIM2}_${MODEL}_${CDO_COMMAND_1}${SUFFIX}.nc
    cdo sub ${CMOR_VAR}_${SIM1}_${MODEL}_${CDO_COMMAND_1}${SUFFIX}.nc ${CMOR_VAR}_${SIM2}_${MODEL}_${CDO_COMMAND_1}${SUFFIX}.nc ${CMOR_VAR}_${SIM1}-${SIM2}_${MODEL}_${CDO_COMMAND_1}${SUFFIX}.nc
    cp ${CMOR_VAR}_${SIM1}-${SIM2}_${MODEL}_${CDO_COMMAND_1}${SUFFIX}.nc ${CMOR_VAR}_${SIM1}-${SIM2}_${MODEL}_${CDO_COMMAND_1}${SUFFIX}_signal-separated.nc
    NC_FILE=${SCRATCH}/${SCENARIO_COMBNATION}/${CMOR_TABLE}/${CMOR_VAR}/${CMOR_VAR}_${SIM1}-${SIM2}_${MODEL}_${CDO_COMMAND_1}${SUFFIX}_signal-separated.nc
    
    echo "Calculate signal separation for variable ${CMOR_VAR} in file ${NC_FILE}"
#    python /pf/b/b380948/lamaclima/prepare_local_nonlocal_total_effects.py ${NC_FILE} ${CMOR_VAR} interpolate_local
    python /pf/b/b381334/signal_seperation/prepare_local_nonlocal_total_effects.py ${NC_FILE} ${CMOR_VAR} interpolate_total
    
    else
      # THE SIGNAL SEPARATED FILE DOES ALREADY EXIST AS "multi-year_mean_signal" ALREADY RAN BEFORE
      # ONLY DEFINE THE NC_FILE VARIABLE
      NC_FILE=${SCRATCH}/${SCENARIO_COMBNATION}/${CMOR_TABLE}/${CMOR_VAR}/${CMOR_VAR}_${SIM1}-${SIM2}_${MODEL}_${CDO_COMMAND_1}${SUFFIX}_signal-separated.nc
    
    fi
    
    if [ "${METHOD}" ==  "absolute_values_ctl" ]; then
    
      # ABSOLUTE VALUES FOR TOTAL/LOCAL/NONLOCAL EFFECT: time_series(CTL) + multi_year_mean(delta total/local/nonlocal)
      cdo ${CDO_COMMAND_2_ADD} -setname,absolute_${CMOR_VAR}_total ${CMOR_VAR}_${SIM2}_${MODEL}${SUFFIX}.nc -selname,${CMOR_VAR}_total ${NC_FILE} ABS_TOTAL_TEMPORAL.nc
      cdo ${CDO_COMMAND_2_ADD} -setname,absolute_${CMOR_VAR}_local ${CMOR_VAR}_${SIM2}_${MODEL}${SUFFIX}.nc -selname,${CMOR_VAR}_local ${NC_FILE}  ABS_LOCAL_TEMPORAL.nc
      cdo ${CDO_COMMAND_2_ADD} -setname,absolute_${CMOR_VAR}_nonlocal ${CMOR_VAR}_${SIM2}_${MODEL}${SUFFIX}.nc -selname,${CMOR_VAR}_nonlocal ${NC_FILE}  ABS_NONLOCAL_TEMPORAL.nc    
    
      # CREATE ONE FILE WITH ABSOLUTE VALUES FOR ALL EFFECTS BASED ON CTL CLIMATE
      cdo merge ABS_TOTAL_TEMPORAL.nc ABS_LOCAL_TEMPORAL.nc ABS_NONLOCAL_TEMPORAL.nc ${CMOR_VAR}_${SIM1}_${MODEL}_signal-separated_absolute_values_${SIM2}-climate_${SUFFIX}.nc
      rm ABS_TOTAL_TEMPORAL.nc ABS_LOCAL_TEMPORAL.nc ABS_NONLOCAL_TEMPORAL.nc
    
      # TO CHECK THE EQUATION
      # NC_FILE2=${SCRATCH}/${SCENARIO_COMBNATION}/${CMOR_TABLE}/${CMOR_VAR}/${CMOR_VAR}_${SIM1}-${SIM2}_${MODEL}_${CDO_COMMAND}_signal-separated.nc
    
    fi
  done

# absolute_values_scen: absolute values based on scen time series  
elif [ "${METHOD}" ==  "absolute_values_scen" ]; then
  # absolute values for SCEN_nonlocal and SCEN_total by interpolation of non-LCLM grid points and LCLM grid points, respectively  
  # absolute value for SCEN_local = SCEN_total - SCEN_nonlocal + CTL with CTL as multi-year mean values and SCEN as full time series
  for SUFFIX in {000000..000004}; do
  
    cp ${CMOR_VAR}_${SIM1}_${MODEL}${SUFFIX}.nc ${CMOR_VAR}_${SIM1}_${MODEL}_signal-separated_absolute_values_${SIM1}-climate_false-local_${SUFFIX}.nc
    NC_FILE=${SCRATCH}/${SCENARIO_COMBNATION}/${CMOR_TABLE}/${CMOR_VAR}/${CMOR_VAR}_${SIM1}_${MODEL}_signal-separated_absolute_values_${SIM1}-climate_false-local_${SUFFIX}.nc
  
    echo "Calculate signal separation for variable ${CMOR_VAR} in file ${NC_FILE}"
    # METHOD "interpolate_total" NEEDED TO CORRECTLY CALCULATE TOTAL CLIMATE W/O USING FALSE LOCAL INFORMATION
    # FOR ABSOLUTE VALUES THE EQUATION total = local + nonlocal MUST NOT BE APPLIED!!
    python /pf/b/b381334/signal_seperation/prepare_local_nonlocal_total_effects.py ${NC_FILE} ${CMOR_VAR} interpolate_total
    
    # SCEN_local = SCEN_total - SCEN_nonlocal + CTL; with CTL as multi-year mean values and SCEN as full time series
    # fh: separated long command into two shorter
    # cdo ${CDO_COMMAND_2_ADD} -sub -selname,${CMOR_VAR}_total ${NC_FILE} -selname,${CMOR_VAR}_nonlocal ${NC_FILE} ${CMOR_VAR}_${SIM2}_${MODEL}_${CDO_COMMAND_1}${SUFFIX}.nc ABS_LOCAL_TEMPORAL.nc
#fh not needed     cdo sub -selname,${CMOR_VAR}_total ${NC_FILE} -selname,${CMOR_VAR}_nonlocal ${NC_FILE} ABS_TOTAL-NONLOCAL_TEMPORAL.nc
#fh not needed     cdo ${CDO_COMMAND_2_ADD} ABS_TOTAL-NONLOCAL_TEMPORAL.nc ${CMOR_VAR}_${SIM2}_${MODEL}_${CDO_COMMAND_1}${SUFFIX}.nc ABS_LOCAL_TEMPORAL.nc        
    cdo ${CDO_COMMAND_1} ${CMOR_VAR}_${SIM2}_${MODEL}${SUFFIX}.nc ${CMOR_VAR}_${SIM2}_${MODEL}_${CDO_COMMAND_1}${SUFFIX}.nc
    cdo ${CDO_COMMAND_2_ADD} -selname,${CMOR_VAR}_local ${NC_FILE} ${CMOR_VAR}_${SIM2}_${MODEL}_${CDO_COMMAND_1}${SUFFIX}.nc ABS_LOCAL_TEMPORAL.nc
    
    # RENAMING 
    cdo setname,absolute_${CMOR_VAR}_local ABS_LOCAL_TEMPORAL.nc ABS_LOCAL_TEMPORAL_RENAME.nc
    rm ABS_LOCAL_TEMPORAL.nc
    
    # MERGING
    cdo merge -setname,absolute_${CMOR_VAR}_nonlocal -selname,${CMOR_VAR}_nonlocal ${NC_FILE} -setname,absolute_${CMOR_VAR}_total -selname,${CMOR_VAR}_total ${NC_FILE} ABS_LOCAL_TEMPORAL_RENAME.nc ${CMOR_VAR}_${SIM1}_${MODEL}_signal-separated_absolute_values_${SIM1}-climate_${SUFFIX}.nc
    rm ABS_LOCAL_TEMPORAL_RENAME.nc ${NC_FILE}
    
  done
    
fi
done
