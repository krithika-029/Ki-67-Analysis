#!/bin/bash
# TUPAC16 Quick Registration and Download Guide

echo "🏆 TUPAC16 REGISTRATION & DOWNLOAD HELPER"
echo "=========================================="
echo ""

echo "🌐 STEP 1: REGISTER (Browser opened automatically)"
echo "   Visit: http://tupac.tue-image.nl/"
echo "   • Click 'Register' or 'Download Data'"
echo "   • Fill in your details:"
echo "     - Email address"
echo "     - Institution/University"
echo "     - Research purpose (Ki-67 analysis)"
echo "   • Verify your email"
echo ""

echo "📥 STEP 2: DOWNLOAD LINKS YOU'LL RECEIVE"
echo "   After registration, you'll get download links for:"
echo "   • Training images (TUPAC16_training.zip)"
echo "   • Test images (TUPAC16_test.zip)" 
echo "   • Annotations (TUPAC16_annotations.zip)"
echo "   • Evaluation kit (TUPAC16_evaluation.zip)"
echo ""

echo "💾 STEP 3: DOWNLOAD COMMANDS"
echo "   Once you get the download links, use:"
echo "   wget '<training_link>' -O TUPAC16_training.zip"
echo "   wget '<test_link>' -O TUPAC16_test.zip"
echo "   wget '<annotations_link>' -O TUPAC16_annotations.zip"
echo ""

echo "📁 STEP 4: EXTRACT FILES"
echo "   unzip TUPAC16_training.zip -d TUPAC16_Dataset/images/training/"
echo "   unzip TUPAC16_test.zip -d TUPAC16_Dataset/images/test/"
echo "   unzip TUPAC16_annotations.zip -d TUPAC16_Dataset/annotations/"
echo ""

echo "🧪 STEP 5: TEST DATASET"
echo "   python tupac16_dataset.py"
echo "   python tupac16_comparison.py"
echo ""

echo "⚡ ALTERNATIVE: QUICK START WITH BREAKHIS"
echo "   If TUPAC16 registration takes time, start with BreakHis:"
echo "   python quick_dataset_setup.py"
echo ""

echo "📧 EXPECTED REGISTRATION EMAIL:"
echo "   Subject: TUPAC16 Dataset Access"
echo "   Content: Download links + instructions"
echo "   Time: Usually within 24 hours"
echo ""

echo "🎯 WHY TUPAC16 > FILE SIZE METHOD:"
echo "   ✅ Expert pathologist ground truth"
echo "   ✅ Real Ki-67 proliferation scores"
echo "   ✅ Standardized clinical protocol"
echo "   ✅ International competition benchmark"
echo "   ❌ Your current method uses arbitrary file sizes"
echo ""

echo "🚀 QUICK STATUS CHECK:"
if [ -d "TUPAC16_Dataset/images/training" ] && [ "$(ls -A TUPAC16_Dataset/images/training 2>/dev/null)" ]; then
    echo "   ✅ TUPAC16 training images found!"
    echo "   📊 Image count: $(find TUPAC16_Dataset/images/training -name "*.tif*" | wc -l)"
else
    echo "   ⏳ TUPAC16 not downloaded yet"
    echo "   👆 Follow registration steps above"
fi

echo ""
echo "💡 NEED IMMEDIATE ALTERNATIVE?"
echo "   Download BreakHis (no registration): python quick_dataset_setup.py"
echo "   7,909 breast cancer images with proper annotations!"
