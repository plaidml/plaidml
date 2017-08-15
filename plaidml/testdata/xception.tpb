code: "function (\n  X_I_0[X_I_0_0, X_I_0_1],\n  X_I_1[X_I_1_0, X_I_1_1],\n  X_I_2[X_I_2_0, X_I_2_1, X_I_2_2, X_I_2_3],\n  X_I_3[X_I_3_0, X_I_3_1, X_I_3_2, X_I_3_3],\n  X_I_4[X_I_4_0],\n  X_I_5[X_I_5_0],\n  X_I_6[X_I_6_0],\n  X_I_7[X_I_7_0],\n  X_I_8[X_I_8_0, X_I_8_1],\n  X_I_9[X_I_9_0, X_I_9_1, X_I_9_2, X_I_9_3],\n  X_I_10[X_I_10_0],\n  X_I_11[X_I_11_0],\n  X_I_12[X_I_12_0],\n  X_I_13[X_I_13_0],\n  X_I_14[X_I_14_0, X_I_14_1, X_I_14_2, X_I_14_3],\n  X_I_15[X_I_15_0, X_I_15_1, X_I_15_2, X_I_15_3],\n  X_I_16[X_I_16_0],\n  X_I_17[X_I_17_0],\n  X_I_18[X_I_18_0],\n  X_I_19[X_I_19_0],\n  X_I_20[X_I_20_0, X_I_20_1, X_I_20_2, X_I_20_3],\n  X_I_21[X_I_21_0, X_I_21_1, X_I_21_2, X_I_21_3],\n  X_I_22[X_I_22_0],\n  X_I_23[X_I_23_0],\n  X_I_24[X_I_24_0],\n  X_I_25[X_I_25_0],\n  X_I_26[X_I_26_0, X_I_26_1, X_I_26_2, X_I_26_3],\n  X_I_27[X_I_27_0],\n  X_I_28[X_I_28_0],\n  X_I_29[X_I_29_0],\n  X_I_30[X_I_30_0],\n  X_I_31[X_I_31_0, X_I_31_1, X_I_31_2, X_I_31_3],\n  X_I_32[X_I_32_0, X_I_32_1, X_I_32_2, X_I_32_3],\n  X_I_33[X_I_33_0],\n  X_I_34[X_I_34_0],\n  X_I_35[X_I_35_0],\n  X_I_36[X_I_36_0],\n  X_I_37[X_I_37_0, X_I_37_1, X_I_37_2, X_I_37_3],\n  X_I_38[X_I_38_0, X_I_38_1, X_I_38_2, X_I_38_3],\n  X_I_39[X_I_39_0],\n  X_I_40[X_I_40_0],\n  X_I_41[X_I_41_0],\n  X_I_42[X_I_42_0],\n  X_I_43[X_I_43_0, X_I_43_1, X_I_43_2, X_I_43_3],\n  X_I_44[X_I_44_0],\n  X_I_45[X_I_45_0],\n  X_I_46[X_I_46_0],\n  X_I_47[X_I_47_0],\n  X_I_48[X_I_48_0, X_I_48_1, X_I_48_2, X_I_48_3],\n  X_I_49[X_I_49_0, X_I_49_1, X_I_49_2, X_I_49_3],\n  X_I_50[X_I_50_0],\n  X_I_51[X_I_51_0],\n  X_I_52[X_I_52_0],\n  X_I_53[X_I_53_0],\n  X_I_54[X_I_54_0, X_I_54_1, X_I_54_2, X_I_54_3],\n  X_I_55[X_I_55_0, X_I_55_1, X_I_55_2, X_I_55_3],\n  X_I_56[X_I_56_0],\n  X_I_57[X_I_57_0],\n  X_I_58[X_I_58_0],\n  X_I_59[X_I_59_0],\n  X_I_60[X_I_60_0, X_I_60_1, X_I_60_2, X_I_60_3],\n  X_I_61[X_I_61_0],\n  X_I_62[X_I_62_0],\n  X_I_63[X_I_63_0],\n  X_I_64[X_I_64_0],\n  X_I_65[X_I_65_0, X_I_65_1, X_I_65_2, X_I_65_3],\n  X_I_66[X_I_66_0, X_I_66_1, X_I_66_2, X_I_66_3],\n  X_I_67[X_I_67_0],\n  X_I_68[X_I_68_0],\n  X_I_69[X_I_69_0],\n  X_I_70[X_I_70_0],\n  X_I_71[X_I_71_0, X_I_71_1, X_I_71_2, X_I_71_3],\n  X_I_72[X_I_72_0, X_I_72_1, X_I_72_2, X_I_72_3],\n  X_I_73[X_I_73_0],\n  X_I_74[X_I_74_0],\n  X_I_75[X_I_75_0],\n  X_I_76[X_I_76_0],\n  X_I_77[X_I_77_0, X_I_77_1, X_I_77_2, X_I_77_3],\n  X_I_78[X_I_78_0, X_I_78_1, X_I_78_2, X_I_78_3],\n  X_I_79[X_I_79_0],\n  X_I_80[X_I_80_0],\n  X_I_81[X_I_81_0],\n  X_I_82[X_I_82_0],\n  X_I_83[X_I_83_0, X_I_83_1, X_I_83_2, X_I_83_3],\n  X_I_84[X_I_84_0, X_I_84_1, X_I_84_2, X_I_84_3],\n  X_I_85[X_I_85_0],\n  X_I_86[X_I_86_0],\n  X_I_87[X_I_87_0],\n  X_I_88[X_I_88_0],\n  X_I_89[X_I_89_0, X_I_89_1, X_I_89_2, X_I_89_3],\n  X_I_90[X_I_90_0, X_I_90_1, X_I_90_2, X_I_90_3],\n  X_I_91[X_I_91_0],\n  X_I_92[X_I_92_0],\n  X_I_93[X_I_93_0],\n  X_I_94[X_I_94_0],\n  X_I_95[X_I_95_0, X_I_95_1, X_I_95_2, X_I_95_3],\n  X_I_96[X_I_96_0, X_I_96_1, X_I_96_2, X_I_96_3],\n  X_I_97[X_I_97_0],\n  X_I_98[X_I_98_0],\n  X_I_99[X_I_99_0],\n  X_I_100[X_I_100_0],\n  X_I_101[X_I_101_0, X_I_101_1, X_I_101_2, X_I_101_3],\n  X_I_102[X_I_102_0, X_I_102_1, X_I_102_2, X_I_102_3],\n  X_I_103[X_I_103_0],\n  X_I_104[X_I_104_0],\n  X_I_105[X_I_105_0],\n  X_I_106[X_I_106_0],\n  X_I_107[X_I_107_0, X_I_107_1, X_I_107_2, X_I_107_3],\n  X_I_108[X_I_108_0, X_I_108_1, X_I_108_2, X_I_108_3],\n  X_I_109[X_I_109_0],\n  X_I_110[X_I_110_0],\n  X_I_111[X_I_111_0],\n  X_I_112[X_I_112_0],\n  X_I_113[X_I_113_0, X_I_113_1, X_I_113_2, X_I_113_3],\n  X_I_114[X_I_114_0, X_I_114_1, X_I_114_2, X_I_114_3],\n  X_I_115[X_I_115_0],\n  X_I_116[X_I_116_0],\n  X_I_117[X_I_117_0],\n  X_I_118[X_I_118_0],\n  X_I_119[X_I_119_0, X_I_119_1, X_I_119_2, X_I_119_3],\n  X_I_120[X_I_120_0, X_I_120_1, X_I_120_2, X_I_120_3],\n  X_I_121[X_I_121_0],\n  X_I_122[X_I_122_0],\n  X_I_123[X_I_123_0],\n  X_I_124[X_I_124_0],\n  X_I_125[X_I_125_0, X_I_125_1, X_I_125_2, X_I_125_3],\n  X_I_126[X_I_126_0, X_I_126_1, X_I_126_2, X_I_126_3],\n  X_I_127[X_I_127_0],\n  X_I_128[X_I_128_0],\n  X_I_129[X_I_129_0],\n  X_I_130[X_I_130_0],\n  X_I_131[X_I_131_0, X_I_131_1, X_I_131_2, X_I_131_3],\n  X_I_132[X_I_132_0, X_I_132_1, X_I_132_2, X_I_132_3],\n  X_I_133[X_I_133_0],\n  X_I_134[X_I_134_0],\n  X_I_135[X_I_135_0],\n  X_I_136[X_I_136_0],\n  X_I_137[X_I_137_0, X_I_137_1, X_I_137_2, X_I_137_3],\n  X_I_138[X_I_138_0, X_I_138_1, X_I_138_2, X_I_138_3],\n  X_I_139[X_I_139_0],\n  X_I_140[X_I_140_0],\n  X_I_141[X_I_141_0],\n  X_I_142[X_I_142_0],\n  X_I_143[X_I_143_0, X_I_143_1, X_I_143_2, X_I_143_3],\n  X_I_144[X_I_144_0, X_I_144_1, X_I_144_2, X_I_144_3],\n  X_I_145[X_I_145_0],\n  X_I_146[X_I_146_0],\n  X_I_147[X_I_147_0],\n  X_I_148[X_I_148_0],\n  X_I_149[X_I_149_0, X_I_149_1, X_I_149_2, X_I_149_3],\n  X_I_150[X_I_150_0, X_I_150_1, X_I_150_2, X_I_150_3],\n  X_I_151[X_I_151_0],\n  X_I_152[X_I_152_0],\n  X_I_153[X_I_153_0],\n  X_I_154[X_I_154_0],\n  X_I_155[X_I_155_0, X_I_155_1, X_I_155_2, X_I_155_3],\n  X_I_156[X_I_156_0, X_I_156_1, X_I_156_2, X_I_156_3],\n  X_I_157[X_I_157_0],\n  X_I_158[X_I_158_0],\n  X_I_159[X_I_159_0],\n  X_I_160[X_I_160_0],\n  X_I_161[X_I_161_0, X_I_161_1, X_I_161_2, X_I_161_3],\n  X_I_162[X_I_162_0, X_I_162_1, X_I_162_2, X_I_162_3],\n  X_I_163[X_I_163_0],\n  X_I_164[X_I_164_0],\n  X_I_165[X_I_165_0],\n  X_I_166[X_I_166_0],\n  X_I_167[X_I_167_0, X_I_167_1, X_I_167_2, X_I_167_3],\n  X_I_168[X_I_168_0, X_I_168_1, X_I_168_2, X_I_168_3],\n  X_I_169[X_I_169_0],\n  X_I_170[X_I_170_0],\n  X_I_171[X_I_171_0],\n  X_I_172[X_I_172_0],\n  X_I_173[X_I_173_0, X_I_173_1, X_I_173_2, X_I_173_3],\n  X_I_174[X_I_174_0, X_I_174_1, X_I_174_2, X_I_174_3],\n  X_I_175[X_I_175_0],\n  X_I_176[X_I_176_0],\n  X_I_177[X_I_177_0],\n  X_I_178[X_I_178_0],\n  X_I_179[X_I_179_0, X_I_179_1, X_I_179_2, X_I_179_3],\n  X_I_180[X_I_180_0, X_I_180_1, X_I_180_2, X_I_180_3],\n  X_I_181[X_I_181_0],\n  X_I_182[X_I_182_0],\n  X_I_183[X_I_183_0],\n  X_I_184[X_I_184_0],\n  X_I_185[X_I_185_0, X_I_185_1, X_I_185_2, X_I_185_3],\n  X_I_186[X_I_186_0, X_I_186_1, X_I_186_2, X_I_186_3],\n  X_I_187[X_I_187_0],\n  X_I_188[X_I_188_0],\n  X_I_189[X_I_189_0],\n  X_I_190[X_I_190_0],\n  X_I_191[X_I_191_0, X_I_191_1, X_I_191_2, X_I_191_3],\n  X_I_192[X_I_192_0, X_I_192_1, X_I_192_2, X_I_192_3],\n  X_I_193[X_I_193_0],\n  X_I_194[X_I_194_0],\n  X_I_195[X_I_195_0],\n  X_I_196[X_I_196_0],\n  X_I_197[X_I_197_0, X_I_197_1, X_I_197_2, X_I_197_3],\n  X_I_198[X_I_198_0, X_I_198_1, X_I_198_2, X_I_198_3],\n  X_I_199[X_I_199_0],\n  X_I_200[X_I_200_0],\n  X_I_201[X_I_201_0],\n  X_I_202[X_I_202_0],\n  X_I_203[X_I_203_0, X_I_203_1, X_I_203_2, X_I_203_3],\n  X_I_204[X_I_204_0, X_I_204_1, X_I_204_2, X_I_204_3],\n  X_I_205[X_I_205_0],\n  X_I_206[X_I_206_0],\n  X_I_207[X_I_207_0],\n  X_I_208[X_I_208_0],\n  X_I_209[X_I_209_0, X_I_209_1, X_I_209_2, X_I_209_3],\n  X_I_210[X_I_210_0, X_I_210_1, X_I_210_2, X_I_210_3],\n  X_I_211[X_I_211_0],\n  X_I_212[X_I_212_0],\n  X_I_213[X_I_213_0],\n  X_I_214[X_I_214_0],\n  X_I_215[X_I_215_0, X_I_215_1, X_I_215_2, X_I_215_3],\n  X_I_216[X_I_216_0, X_I_216_1, X_I_216_2, X_I_216_3],\n  X_I_217[X_I_217_0],\n  X_I_218[X_I_218_0],\n  X_I_219[X_I_219_0],\n  X_I_220[X_I_220_0],\n  X_I_221[X_I_221_0, X_I_221_1, X_I_221_2, X_I_221_3],\n  X_I_222[X_I_222_0],\n  X_I_223[X_I_223_0],\n  X_I_224[X_I_224_0],\n  X_I_225[X_I_225_0],\n  X_I_226[X_I_226_0, X_I_226_1, X_I_226_2, X_I_226_3],\n  X_I_227[X_I_227_0, X_I_227_1, X_I_227_2, X_I_227_3],\n  X_I_228[X_I_228_0],\n  X_I_229[X_I_229_0],\n  X_I_230[X_I_230_0],\n  X_I_231[X_I_231_0],\n  X_I_232[X_I_232_0, X_I_232_1, X_I_232_2, X_I_232_3],\n  X_I_233[X_I_233_0, X_I_233_1, X_I_233_2, X_I_233_3],\n  X_I_234[X_I_234_0],\n  X_I_235[X_I_235_0],\n  X_I_236[X_I_236_0],\n  X_I_237[X_I_237_0],\n  X_I_238[X_I_238_0, X_I_238_1],\n  X_I_239[X_I_239_0]\n) -> (\n  X_T943\n) {\n  X_T6 = 299;\n  X_T7 = 3;\n  X_T8 = sub(X_T6, X_T7);\n  X_T9 = 2;\n  X_T10 = add(X_T8, X_T9);\n  X_T11 = div(X_T10, X_T9);\n  X_T12 = sub(X_T11, X_T7);\n  X_T13 = 1;\n  X_T14 = add(X_T12, X_T13);\n  X_T15 = div(X_T14, X_T13);\n  X_T16 = add(X_T15, X_T13);\n  X_T17 = sub(X_T16, X_T13);\n  X_T18 = div(X_T17, X_T13);\n  X_T19 = sub(X_T18, X_T13);\n  X_T20 = add(X_T19, X_T13);\n  X_T21 = div(X_T20, X_T13);\n  X_T22 = add(X_T21, X_T13);\n  X_T23 = sub(X_T22, X_T13);\n  X_T24 = div(X_T23, X_T13);\n  X_T25 = sub(X_T24, X_T13);\n  X_T26 = add(X_T25, X_T13);\n  X_T27 = div(X_T26, X_T13);\n  X_T28 = add(X_T27, X_T9);\n  X_T29 = sub(X_T28, X_T13);\n  X_T30 = div(X_T29, X_T9);\n  X_T31 = add(X_T15, X_T9);\n  X_T32 = sub(X_T31, X_T13);\n  X_T33 = div(X_T32, X_T9);\n  X_T34 = broadcast(X_T30, X_T33);\n  X_T35 = add(X_T34, X_T13);\n  X_T36 = sub(X_T35, X_T13);\n  X_T37 = div(X_T36, X_T13);\n  X_T38 = sub(X_T37, X_T13);\n  X_T39 = add(X_T38, X_T13);\n  X_T40 = div(X_T39, X_T13);\n  X_T41 = add(X_T40, X_T13);\n  X_T42 = sub(X_T41, X_T13);\n  X_T43 = div(X_T42, X_T13);\n  X_T44 = sub(X_T43, X_T13);\n  X_T45 = add(X_T44, X_T13);\n  X_T46 = div(X_T45, X_T13);\n  X_T47 = add(X_T46, X_T9);\n  X_T48 = sub(X_T47, X_T13);\n  X_T49 = div(X_T48, X_T9);\n  X_T50 = add(X_T34, X_T9);\n  X_T51 = sub(X_T50, X_T13);\n  X_T52 = div(X_T51, X_T9);\n  X_T53 = broadcast(X_T49, X_T52);\n  X_T54 = add(X_T53, X_T13);\n  X_T55 = sub(X_T54, X_T13);\n  X_T56 = div(X_T55, X_T13);\n  X_T57 = sub(X_T56, X_T13);\n  X_T58 = add(X_T57, X_T13);\n  X_T59 = div(X_T58, X_T13);\n  X_T60 = add(X_T59, X_T13);\n  X_T61 = sub(X_T60, X_T13);\n  X_T62 = div(X_T61, X_T13);\n  X_T63 = sub(X_T62, X_T13);\n  X_T64 = add(X_T63, X_T13);\n  X_T65 = div(X_T64, X_T13);\n  X_T66 = add(X_T65, X_T9);\n  X_T67 = sub(X_T66, X_T13);\n  X_T68 = div(X_T67, X_T9);\n  X_T69 = add(X_T53, X_T9);\n  X_T70 = sub(X_T69, X_T13);\n  X_T71 = div(X_T70, X_T9);\n  X_T72 = broadcast(X_T68, X_T71);\n  X_T73 = add(X_T72, X_T13);\n  X_T74 = sub(X_T73, X_T13);\n  X_T75 = div(X_T74, X_T13);\n  X_T76 = sub(X_T75, X_T13);\n  X_T77 = add(X_T76, X_T13);\n  X_T78 = div(X_T77, X_T13);\n  X_T79 = add(X_T78, X_T13);\n  X_T80 = sub(X_T79, X_T13);\n  X_T81 = div(X_T80, X_T13);\n  X_T82 = sub(X_T81, X_T13);\n  X_T83 = add(X_T82, X_T13);\n  X_T84 = div(X_T83, X_T13);\n  X_T85 = add(X_T84, X_T13);\n  X_T86 = sub(X_T85, X_T13);\n  X_T87 = div(X_T86, X_T13);\n  X_T88 = sub(X_T87, X_T13);\n  X_T89 = add(X_T88, X_T13);\n  X_T90 = div(X_T89, X_T13);\n  X_T91 = broadcast(X_T90, X_T72);\n  X_T92 = add(X_T91, X_T13);\n  X_T93 = sub(X_T92, X_T13);\n  X_T94 = div(X_T93, X_T13);\n  X_T95 = sub(X_T94, X_T13);\n  X_T96 = add(X_T95, X_T13);\n  X_T97 = div(X_T96, X_T13);\n  X_T98 = add(X_T97, X_T13);\n  X_T99 = sub(X_T98, X_T13);\n  X_T100 = div(X_T99, X_T13);\n  X_T101 = sub(X_T100, X_T13);\n  X_T102 = add(X_T101, X_T13);\n  X_T103 = div(X_T102, X_T13);\n  X_T104 = add(X_T103, X_T13);\n  X_T105 = sub(X_T104, X_T13);\n  X_T106 = div(X_T105, X_T13);\n  X_T107 = sub(X_T106, X_T13);\n  X_T108 = add(X_T107, X_T13);\n  X_T109 = div(X_T108, X_T13);\n  X_T110 = broadcast(X_T109, X_T91);\n  X_T111 = add(X_T110, X_T13);\n  X_T112 = sub(X_T111, X_T13);\n  X_T113 = div(X_T112, X_T13);\n  X_T114 = sub(X_T113, X_T13);\n  X_T115 = add(X_T114, X_T13);\n  X_T116 = div(X_T115, X_T13);\n  X_T117 = add(X_T116, X_T13);\n  X_T118 = sub(X_T117, X_T13);\n  X_T119 = div(X_T118, X_T13);\n  X_T120 = sub(X_T119, X_T13);\n  X_T121 = add(X_T120, X_T13);\n  X_T122 = div(X_T121, X_T13);\n  X_T123 = add(X_T122, X_T13);\n  X_T124 = sub(X_T123, X_T13);\n  X_T125 = div(X_T124, X_T13);\n  X_T126 = sub(X_T125, X_T13);\n  X_T127 = add(X_T126, X_T13);\n  X_T128 = div(X_T127, X_T13);\n  X_T129 = broadcast(X_T128, X_T110);\n  X_T130 = add(X_T129, X_T13);\n  X_T131 = sub(X_T130, X_T13);\n  X_T132 = div(X_T131, X_T13);\n  X_T133 = sub(X_T132, X_T13);\n  X_T134 = add(X_T133, X_T13);\n  X_T135 = div(X_T134, X_T13);\n  X_T136 = add(X_T135, X_T13);\n  X_T137 = sub(X_T136, X_T13);\n  X_T138 = div(X_T137, X_T13);\n  X_T139 = sub(X_T138, X_T13);\n  X_T140 = add(X_T139, X_T13);\n  X_T141 = div(X_T140, X_T13);\n  X_T142 = add(X_T141, X_T13);\n  X_T143 = sub(X_T142, X_T13);\n  X_T144 = div(X_T143, X_T13);\n  X_T145 = sub(X_T144, X_T13);\n  X_T146 = add(X_T145, X_T13);\n  X_T147 = div(X_T146, X_T13);\n  X_T148 = broadcast(X_T147, X_T129);\n  X_T149 = add(X_T148, X_T13);\n  X_T150 = sub(X_T149, X_T13);\n  X_T151 = div(X_T150, X_T13);\n  X_T152 = sub(X_T151, X_T13);\n  X_T153 = add(X_T152, X_T13);\n  X_T154 = div(X_T153, X_T13);\n  X_T155 = add(X_T154, X_T13);\n  X_T156 = sub(X_T155, X_T13);\n  X_T157 = div(X_T156, X_T13);\n  X_T158 = sub(X_T157, X_T13);\n  X_T159 = add(X_T158, X_T13);\n  X_T160 = div(X_T159, X_T13);\n  X_T161 = add(X_T160, X_T13);\n  X_T162 = sub(X_T161, X_T13);\n  X_T163 = div(X_T162, X_T13);\n  X_T164 = sub(X_T163, X_T13);\n  X_T165 = add(X_T164, X_T13);\n  X_T166 = div(X_T165, X_T13);\n  X_T167 = broadcast(X_T166, X_T148);\n  X_T168 = add(X_T167, X_T13);\n  X_T169 = sub(X_T168, X_T13);\n  X_T170 = div(X_T169, X_T13);\n  X_T171 = sub(X_T170, X_T13);\n  X_T172 = add(X_T171, X_T13);\n  X_T173 = div(X_T172, X_T13);\n  X_T174 = add(X_T173, X_T13);\n  X_T175 = sub(X_T174, X_T13);\n  X_T176 = div(X_T175, X_T13);\n  X_T177 = sub(X_T176, X_T13);\n  X_T178 = add(X_T177, X_T13);\n  X_T179 = div(X_T178, X_T13);\n  X_T180 = add(X_T179, X_T13);\n  X_T181 = sub(X_T180, X_T13);\n  X_T182 = div(X_T181, X_T13);\n  X_T183 = sub(X_T182, X_T13);\n  X_T184 = add(X_T183, X_T13);\n  X_T185 = div(X_T184, X_T13);\n  X_T186 = broadcast(X_T185, X_T167);\n  X_T187 = add(X_T186, X_T13);\n  X_T188 = sub(X_T187, X_T13);\n  X_T189 = div(X_T188, X_T13);\n  X_T190 = sub(X_T189, X_T13);\n  X_T191 = add(X_T190, X_T13);\n  X_T192 = div(X_T191, X_T13);\n  X_T193 = add(X_T192, X_T13);\n  X_T194 = sub(X_T193, X_T13);\n  X_T195 = div(X_T194, X_T13);\n  X_T196 = sub(X_T195, X_T13);\n  X_T197 = add(X_T196, X_T13);\n  X_T198 = div(X_T197, X_T13);\n  X_T199 = add(X_T198, X_T13);\n  X_T200 = sub(X_T199, X_T13);\n  X_T201 = div(X_T200, X_T13);\n  X_T202 = sub(X_T201, X_T13);\n  X_T203 = add(X_T202, X_T13);\n  X_T204 = div(X_T203, X_T13);\n  X_T205 = broadcast(X_T204, X_T186);\n  X_T206 = add(X_T205, X_T13);\n  X_T207 = sub(X_T206, X_T13);\n  X_T208 = div(X_T207, X_T13);\n  X_T209 = sub(X_T208, X_T13);\n  X_T210 = add(X_T209, X_T13);\n  X_T211 = div(X_T210, X_T13);\n  X_T212 = add(X_T211, X_T13);\n  X_T213 = sub(X_T212, X_T13);\n  X_T214 = div(X_T213, X_T13);\n  X_T215 = sub(X_T214, X_T13);\n  X_T216 = add(X_T215, X_T13);\n  X_T217 = div(X_T216, X_T13);\n  X_T218 = add(X_T217, X_T13);\n  X_T219 = sub(X_T218, X_T13);\n  X_T220 = div(X_T219, X_T13);\n  X_T221 = sub(X_T220, X_T13);\n  X_T222 = add(X_T221, X_T13);\n  X_T223 = div(X_T222, X_T13);\n  X_T224 = broadcast(X_T223, X_T205);\n  X_T225 = add(X_T224, X_T13);\n  X_T226 = sub(X_T225, X_T13);\n  X_T227 = div(X_T226, X_T13);\n  X_T228 = sub(X_T227, X_T13);\n  X_T229 = add(X_T228, X_T13);\n  X_T230 = div(X_T229, X_T13);\n  X_T231 = add(X_T230, X_T13);\n  X_T232 = sub(X_T231, X_T13);\n  X_T233 = div(X_T232, X_T13);\n  X_T234 = sub(X_T233, X_T13);\n  X_T235 = add(X_T234, X_T13);\n  X_T236 = div(X_T235, X_T13);\n  X_T237 = add(X_T236, X_T9);\n  X_T238 = sub(X_T237, X_T13);\n  X_T239 = div(X_T238, X_T9);\n  X_T240 = add(X_T224, X_T9);\n  X_T241 = sub(X_T240, X_T13);\n  X_T242 = div(X_T241, X_T9);\n  X_T243 = broadcast(X_T239, X_T242);\n  X_T244 = add(X_T243, X_T13);\n  X_T245 = sub(X_T244, X_T13);\n  X_T246 = div(X_T245, X_T13);\n  X_T247 = sub(X_T246, X_T13);\n  X_T248 = add(X_T247, X_T13);\n  X_T249 = div(X_T248, X_T13);\n  X_T250 = add(X_T249, X_T13);\n  X_T251 = sub(X_T250, X_T13);\n  X_T252 = div(X_T251, X_T13);\n  X_T253 = sub(X_T252, X_T13);\n  X_T254 = add(X_T253, X_T13);\n  X_T255 = div(X_T254, X_T13);\n  X_T345 = 0;\n  X_T344[n, x0, x1, co : 16, 149, 149, 32] = +(X_I_2[n, k0 + 2*x0, k1 + 2*x1, ci] * X_I_3[k0, k1, ci, co]);\n  X_T346 = sub(X_T344, X_I_4);\n  X_T347 = mul(X_T346, X_I_5);\n  X_T348 = 0.001;\n  X_T349 = add(X_I_6, X_T348);\n  X_T350 = cmp_lt(X_T349, X_T345);\n  X_T351 = cond(X_T350, X_T345, X_T349);\n  X_T352 = sqrt(X_T351);\n  X_T353 = div(X_T347, X_T352);\n  X_T354 = add(X_T353, X_I_7);\n  X_T355 = cmp_lt(X_T354, X_T345);\n  X_T356 = 0.0;\n  X_T357 = cond(X_T355, X_T356, X_T354);\n  X_T343[n, i, j, x, y, ci : 16, 6, 6, 37, 37, 32] = +(X_I_1[k, i] * X_T357[n, k + 4*x, j + 4*y, ci]);\n  X_T341[n, i, j, x, y, ci : 16, 6, 6, 37, 37, 32] = +(X_T343[n, i, k, x, y, ci] * X_I_1[k, j]);\n  X_T359[i, j, ci, co : 6, 3, 32, 64] = +(X_I_8[i, k] * X_I_9[k, j, ci, co]);\n  X_T358[i, j, ci, co : 6, 6, 32, 64] = +(X_T359[i, k, ci, co] * X_I_8[j, k]);\n  X_T340[n, i, j, x, y, co : 16, 6, 6, 37, 37, 64] = +(X_T341[n, i, j, x, y, ci] * X_T358[i, j, ci, co]);\n  X_T335[n, i, j, x, y, co : 16, 4, 6, 37, 37, 64] = +(X_I_0[k, i] * X_T340[n, k, j, x, y, co]);\n  X_T333[n, i + 4*x, j + 4*y, co : 16, 147, 147, 64] = +(X_T335[n, i, k, x, y, co] * X_I_0[k, j]) no_defract;\n  X_T360 = sub(X_T333, X_I_10);\n  X_T361 = mul(X_T360, X_I_11);\n  X_T362 = add(X_I_12, X_T348);\n  X_T363 = cmp_lt(X_T362, X_T345);\n  X_T364 = cond(X_T363, X_T345, X_T362);\n  X_T365 = sqrt(X_T364);\n  X_T366 = div(X_T361, X_T365);\n  X_T367 = add(X_T366, X_I_13);\n  X_T368 = cmp_lt(X_T367, X_T345);\n  X_T369 = cond(X_T368, X_T356, X_T367);\n  X_T331[n, x0, x1, c + m : 16, 147, 147, 64] = +(X_T369[n, -1 + k0 + x0, -1 + k1 + x1, c] * X_I_14[k0, k1, c, m]);\n  X_T330[n, x0, x1, co : 16, 147, 147, 128] = +(X_T331[n, k0 + x0, k1 + x1, ci] * X_I_15[k0, k1, ci, co]);\n  X_T374 = sub(X_T330, X_I_16);\n  X_T375 = mul(X_T374, X_I_17);\n  X_T376 = add(X_I_18, X_T348);\n  X_T377 = cmp_lt(X_T376, X_T345);\n  X_T378 = cond(X_T377, X_T345, X_T376);\n  X_T379 = sqrt(X_T378);\n  X_T380 = div(X_T375, X_T379);\n  X_T381 = add(X_T380, X_I_19);\n  X_T382 = cmp_lt(X_T381, X_T345);\n  X_T383 = cond(X_T382, X_T356, X_T381);\n  X_T329[n, x0, x1, c + m : 16, 147, 147, 128] = +(X_T383[n, -1 + k0 + x0, -1 + k1 + x1, c] * X_I_20[k0, k1, c, m]);\n  X_T328[n, x0, x1, co : 16, 147, 147, 128] = +(X_T329[n, k0 + x0, k1 + x1, ci] * X_I_21[k0, k1, ci, co]);\n  X_T388 = sub(X_T328, X_I_22);\n  X_T389 = mul(X_T388, X_I_23);\n  X_T390 = add(X_I_24, X_T348);\n  X_T391 = cmp_lt(X_T390, X_T345);\n  X_T392 = cond(X_T391, X_T345, X_T390);\n  X_T393 = sqrt(X_T392);\n  X_T394 = div(X_T389, X_T393);\n  X_T395 = add(X_T394, X_I_25);\n  X_T327[n, x0, x1, c : 16, 74, 74, 128] = >(X_T395[n, -1 + k0 + 2*x0, -1 + k1 + 2*x1, c]), k0 < 3, k1 < 3;\n  X_T402[n, x0, x1, co : 16, 74, 74, 128] = +(X_T369[n, k0 + 2*x0, k1 + 2*x1, ci] * X_I_26[k0, k1, ci, co]);\n  X_T409 = sub(X_T402, X_I_27);\n  X_T410 = mul(X_T409, X_I_28);\n  X_T411 = add(X_I_29, X_T348);\n  X_T412 = cmp_lt(X_T411, X_T345);\n  X_T413 = cond(X_T412, X_T345, X_T411);\n  X_T414 = sqrt(X_T413);\n  X_T415 = div(X_T410, X_T414);\n  X_T416 = add(X_T415, X_I_30);\n  X_T417 = add(X_T327, X_T416);\n  X_T418 = cmp_lt(X_T417, X_T345);\n  X_T419 = cond(X_T418, X_T356, X_T417);\n  X_T325[n, x0, x1, c + m : 16, 74, 74, 128] = +(X_T419[n, -1 + k0 + x0, -1 + k1 + x1, c] * X_I_31[k0, k1, c, m]);\n  X_T324[n, x0, x1, co : 16, 74, 74, 256] = +(X_T325[n, k0 + x0, k1 + x1, ci] * X_I_32[k0, k1, ci, co]);\n  X_T424 = sub(X_T324, X_I_33);\n  X_T425 = mul(X_T424, X_I_34);\n  X_T426 = add(X_I_35, X_T348);\n  X_T427 = cmp_lt(X_T426, X_T345);\n  X_T428 = cond(X_T427, X_T345, X_T426);\n  X_T429 = sqrt(X_T428);\n  X_T430 = div(X_T425, X_T429);\n  X_T431 = add(X_T430, X_I_36);\n  X_T432 = cmp_lt(X_T431, X_T345);\n  X_T433 = cond(X_T432, X_T356, X_T431);\n  X_T323[n, x0, x1, c + m : 16, 74, 74, 256] = +(X_T433[n, -1 + k0 + x0, -1 + k1 + x1, c] * X_I_37[k0, k1, c, m]);\n  X_T322[n, x0, x1, co : 16, 74, 74, 256] = +(X_T323[n, k0 + x0, k1 + x1, ci] * X_I_38[k0, k1, ci, co]);\n  X_T438 = sub(X_T322, X_I_39);\n  X_T439 = mul(X_T438, X_I_40);\n  X_T440 = add(X_I_41, X_T348);\n  X_T441 = cmp_lt(X_T440, X_T345);\n  X_T442 = cond(X_T441, X_T345, X_T440);\n  X_T443 = sqrt(X_T442);\n  X_T444 = div(X_T439, X_T443);\n  X_T445 = add(X_T444, X_I_42);\n  X_T321[n, x0, x1, c : 16, 37, 37, 256] = >(X_T445[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 3, k1 < 3;\n  X_T452[n, x0, x1, co : 16, 37, 37, 256] = +(X_T417[n, k0 + 2*x0, k1 + 2*x1, ci] * X_I_43[k0, k1, ci, co]);\n  X_T459 = sub(X_T452, X_I_44);\n  X_T460 = mul(X_T459, X_I_45);\n  X_T461 = add(X_I_46, X_T348);\n  X_T462 = cmp_lt(X_T461, X_T345);\n  X_T463 = cond(X_T462, X_T345, X_T461);\n  X_T464 = sqrt(X_T463);\n  X_T465 = div(X_T460, X_T464);\n  X_T466 = add(X_T465, X_I_47);\n  X_T467 = add(X_T321, X_T466);\n  X_T468 = cmp_lt(X_T467, X_T345);\n  X_T469 = cond(X_T468, X_T356, X_T467);\n  X_T319[n, x0, x1, c + m : 16, 37, 37, 256] = +(X_T469[n, -1 + k0 + x0, -1 + k1 + x1, c] * X_I_48[k0, k1, c, m]);\n  X_T318[n, x0, x1, co : 16, 37, 37, 728] = +(X_T319[n, k0 + x0, k1 + x1, ci] * X_I_49[k0, k1, ci, co]);\n  X_T474 = sub(X_T318, X_I_50);\n  X_T475 = mul(X_T474, X_I_51);\n  X_T476 = add(X_I_52, X_T348);\n  X_T477 = cmp_lt(X_T476, X_T345);\n  X_T478 = cond(X_T477, X_T345, X_T476);\n  X_T479 = sqrt(X_T478);\n  X_T480 = div(X_T475, X_T479);\n  X_T481 = add(X_T480, X_I_53);\n  X_T482 = cmp_lt(X_T481, X_T345);\n  X_T483 = cond(X_T482, X_T356, X_T481);\n  X_T317[n, x0, x1, c + m : 16, 37, 37, 728] = +(X_T483[n, -1 + k0 + x0, -1 + k1 + x1, c] * X_I_54[k0, k1, c, m]);\n  X_T316[n, x0, x1, co : 16, 37, 37, 728] = +(X_T317[n, k0 + x0, k1 + x1, ci] * X_I_55[k0, k1, ci, co]);\n  X_T488 = sub(X_T316, X_I_56);\n  X_T489 = mul(X_T488, X_I_57);\n  X_T490 = add(X_I_58, X_T348);\n  X_T491 = cmp_lt(X_T490, X_T345);\n  X_T492 = cond(X_T491, X_T345, X_T490);\n  X_T493 = sqrt(X_T492);\n  X_T494 = div(X_T489, X_T493);\n  X_T495 = add(X_T494, X_I_59);\n  X_T315[n, x0, x1, c : 16, 19, 19, 728] = >(X_T495[n, -1 + k0 + 2*x0, -1 + k1 + 2*x1, c]), k0 < 3, k1 < 3;\n  X_T502[n, x0, x1, co : 16, 19, 19, 728] = +(X_T467[n, k0 + 2*x0, k1 + 2*x1, ci] * X_I_60[k0, k1, ci, co]);\n  X_T509 = sub(X_T502, X_I_61);\n  X_T510 = mul(X_T509, X_I_62);\n  X_T511 = add(X_I_63, X_T348);\n  X_T512 = cmp_lt(X_T511, X_T345);\n  X_T513 = cond(X_T512, X_T345, X_T511);\n  X_T514 = sqrt(X_T513);\n  X_T515 = div(X_T510, X_T514);\n  X_T516 = add(X_T515, X_I_64);\n  X_T517 = add(X_T315, X_T516);\n  X_T518 = cmp_lt(X_T517, X_T345);\n  X_T519 = cond(X_T518, X_T356, X_T517);\n  X_T314[n, x0, x1, c + m : 16, 19, 19, 728] = +(X_T519[n, -1 + k0 + x0, -1 + k1 + x1, c] * X_I_65[k0, k1, c, m]);\n  X_T313[n, x0, x1, co : 16, 19, 19, 728] = +(X_T314[n, k0 + x0, k1 + x1, ci] * X_I_66[k0, k1, ci, co]);\n  X_T524 = sub(X_T313, X_I_67);\n  X_T525 = mul(X_T524, X_I_68);\n  X_T526 = add(X_I_69, X_T348);\n  X_T527 = cmp_lt(X_T526, X_T345);\n  X_T528 = cond(X_T527, X_T345, X_T526);\n  X_T529 = sqrt(X_T528);\n  X_T530 = div(X_T525, X_T529);\n  X_T531 = add(X_T530, X_I_70);\n  X_T532 = cmp_lt(X_T531, X_T345);\n  X_T533 = cond(X_T532, X_T356, X_T531);\n  X_T312[n, x0, x1, c + m : 16, 19, 19, 728] = +(X_T533[n, -1 + k0 + x0, -1 + k1 + x1, c] * X_I_71[k0, k1, c, m]);\n  X_T311[n, x0, x1, co : 16, 19, 19, 728] = +(X_T312[n, k0 + x0, k1 + x1, ci] * X_I_72[k0, k1, ci, co]);\n  X_T538 = sub(X_T311, X_I_73);\n  X_T539 = mul(X_T538, X_I_74);\n  X_T540 = add(X_I_75, X_T348);\n  X_T541 = cmp_lt(X_T540, X_T345);\n  X_T542 = cond(X_T541, X_T345, X_T540);\n  X_T543 = sqrt(X_T542);\n  X_T544 = div(X_T539, X_T543);\n  X_T545 = add(X_T544, X_I_76);\n  X_T546 = cmp_lt(X_T545, X_T345);\n  X_T547 = cond(X_T546, X_T356, X_T545);\n  X_T310[n, x0, x1, c + m : 16, 19, 19, 728] = +(X_T547[n, -1 + k0 + x0, -1 + k1 + x1, c] * X_I_77[k0, k1, c, m]);\n  X_T309[n, x0, x1, co : 16, 19, 19, 728] = +(X_T310[n, k0 + x0, k1 + x1, ci] * X_I_78[k0, k1, ci, co]);\n  X_T552 = sub(X_T309, X_I_79);\n  X_T553 = mul(X_T552, X_I_80);\n  X_T554 = add(X_I_81, X_T348);\n  X_T555 = cmp_lt(X_T554, X_T345);\n  X_T556 = cond(X_T555, X_T345, X_T554);\n  X_T557 = sqrt(X_T556);\n  X_T558 = div(X_T553, X_T557);\n  X_T559 = add(X_T558, X_I_82);\n  X_T560 = add(X_T559, X_T517);\n  X_T561 = cmp_lt(X_T560, X_T345);\n  X_T562 = cond(X_T561, X_T356, X_T560);\n  X_T308[n, x0, x1, c + m : 16, 19, 19, 728] = +(X_T562[n, -1 + k0 + x0, -1 + k1 + x1, c] * X_I_83[k0, k1, c, m]);\n  X_T307[n, x0, x1, co : 16, 19, 19, 728] = +(X_T308[n, k0 + x0, k1 + x1, ci] * X_I_84[k0, k1, ci, co]);\n  X_T567 = sub(X_T307, X_I_85);\n  X_T568 = mul(X_T567, X_I_86);\n  X_T569 = add(X_I_87, X_T348);\n  X_T570 = cmp_lt(X_T569, X_T345);\n  X_T571 = cond(X_T570, X_T345, X_T569);\n  X_T572 = sqrt(X_T571);\n  X_T573 = div(X_T568, X_T572);\n  X_T574 = add(X_T573, X_I_88);\n  X_T575 = cmp_lt(X_T574, X_T345);\n  X_T576 = cond(X_T575, X_T356, X_T574);\n  X_T306[n, x0, x1, c + m : 16, 19, 19, 728] = +(X_T576[n, -1 + k0 + x0, -1 + k1 + x1, c] * X_I_89[k0, k1, c, m]);\n  X_T305[n, x0, x1, co : 16, 19, 19, 728] = +(X_T306[n, k0 + x0, k1 + x1, ci] * X_I_90[k0, k1, ci, co]);\n  X_T581 = sub(X_T305, X_I_91);\n  X_T582 = mul(X_T581, X_I_92);\n  X_T583 = add(X_I_93, X_T348);\n  X_T584 = cmp_lt(X_T583, X_T345);\n  X_T585 = cond(X_T584, X_T345, X_T583);\n  X_T586 = sqrt(X_T585);\n  X_T587 = div(X_T582, X_T586);\n  X_T588 = add(X_T587, X_I_94);\n  X_T589 = cmp_lt(X_T588, X_T345);\n  X_T590 = cond(X_T589, X_T356, X_T588);\n  X_T304[n, x0, x1, c + m : 16, 19, 19, 728] = +(X_T590[n, -1 + k0 + x0, -1 + k1 + x1, c] * X_I_95[k0, k1, c, m]);\n  X_T303[n, x0, x1, co : 16, 19, 19, 728] = +(X_T304[n, k0 + x0, k1 + x1, ci] * X_I_96[k0, k1, ci, co]);\n  X_T595 = sub(X_T303, X_I_97);\n  X_T596 = mul(X_T595, X_I_98);\n  X_T597 = add(X_I_99, X_T348);\n  X_T598 = cmp_lt(X_T597, X_T345);\n  X_T599 = cond(X_T598, X_T345, X_T597);\n  X_T600 = sqrt(X_T599);\n  X_T601 = div(X_T596, X_T600);\n  X_T602 = add(X_T601, X_I_100);\n  X_T603 = add(X_T602, X_T560);\n  X_T604 = cmp_lt(X_T603, X_T345);\n  X_T605 = cond(X_T604, X_T356, X_T603);\n  X_T302[n, x0, x1, c + m : 16, 19, 19, 728] = +(X_T605[n, -1 + k0 + x0, -1 + k1 + x1, c] * X_I_101[k0, k1, c, m]);\n  X_T301[n, x0, x1, co : 16, 19, 19, 728] = +(X_T302[n, k0 + x0, k1 + x1, ci] * X_I_102[k0, k1, ci, co]);\n  X_T610 = sub(X_T301, X_I_103);\n  X_T611 = mul(X_T610, X_I_104);\n  X_T612 = add(X_I_105, X_T348);\n  X_T613 = cmp_lt(X_T612, X_T345);\n  X_T614 = cond(X_T613, X_T345, X_T612);\n  X_T615 = sqrt(X_T614);\n  X_T616 = div(X_T611, X_T615);\n  X_T617 = add(X_T616, X_I_106);\n  X_T618 = cmp_lt(X_T617, X_T345);\n  X_T619 = cond(X_T618, X_T356, X_T617);\n  X_T300[n, x0, x1, c + m : 16, 19, 19, 728] = +(X_T619[n, -1 + k0 + x0, -1 + k1 + x1, c] * X_I_107[k0, k1, c, m]);\n  X_T299[n, x0, x1, co : 16, 19, 19, 728] = +(X_T300[n, k0 + x0, k1 + x1, ci] * X_I_108[k0, k1, ci, co]);\n  X_T624 = sub(X_T299, X_I_109);\n  X_T625 = mul(X_T624, X_I_110);\n  X_T626 = add(X_I_111, X_T348);\n  X_T627 = cmp_lt(X_T626, X_T345);\n  X_T628 = cond(X_T627, X_T345, X_T626);\n  X_T629 = sqrt(X_T628);\n  X_T630 = div(X_T625, X_T629);\n  X_T631 = add(X_T630, X_I_112);\n  X_T632 = cmp_lt(X_T631, X_T345);\n  X_T633 = cond(X_T632, X_T356, X_T631);\n  X_T298[n, x0, x1, c + m : 16, 19, 19, 728] = +(X_T633[n, -1 + k0 + x0, -1 + k1 + x1, c] * X_I_113[k0, k1, c, m]);\n  X_T297[n, x0, x1, co : 16, 19, 19, 728] = +(X_T298[n, k0 + x0, k1 + x1, ci] * X_I_114[k0, k1, ci, co]);\n  X_T638 = sub(X_T297, X_I_115);\n  X_T639 = mul(X_T638, X_I_116);\n  X_T640 = add(X_I_117, X_T348);\n  X_T641 = cmp_lt(X_T640, X_T345);\n  X_T642 = cond(X_T641, X_T345, X_T640);\n  X_T643 = sqrt(X_T642);\n  X_T644 = div(X_T639, X_T643);\n  X_T645 = add(X_T644, X_I_118);\n  X_T646 = add(X_T645, X_T603);\n  X_T647 = cmp_lt(X_T646, X_T345);\n  X_T648 = cond(X_T647, X_T356, X_T646);\n  X_T296[n, x0, x1, c + m : 16, 19, 19, 728] = +(X_T648[n, -1 + k0 + x0, -1 + k1 + x1, c] * X_I_119[k0, k1, c, m]);\n  X_T295[n, x0, x1, co : 16, 19, 19, 728] = +(X_T296[n, k0 + x0, k1 + x1, ci] * X_I_120[k0, k1, ci, co]);\n  X_T653 = sub(X_T295, X_I_121);\n  X_T654 = mul(X_T653, X_I_122);\n  X_T655 = add(X_I_123, X_T348);\n  X_T656 = cmp_lt(X_T655, X_T345);\n  X_T657 = cond(X_T656, X_T345, X_T655);\n  X_T658 = sqrt(X_T657);\n  X_T659 = div(X_T654, X_T658);\n  X_T660 = add(X_T659, X_I_124);\n  X_T661 = cmp_lt(X_T660, X_T345);\n  X_T662 = cond(X_T661, X_T356, X_T660);\n  X_T294[n, x0, x1, c + m : 16, 19, 19, 728] = +(X_T662[n, -1 + k0 + x0, -1 + k1 + x1, c] * X_I_125[k0, k1, c, m]);\n  X_T293[n, x0, x1, co : 16, 19, 19, 728] = +(X_T294[n, k0 + x0, k1 + x1, ci] * X_I_126[k0, k1, ci, co]);\n  X_T667 = sub(X_T293, X_I_127);\n  X_T668 = mul(X_T667, X_I_128);\n  X_T669 = add(X_I_129, X_T348);\n  X_T670 = cmp_lt(X_T669, X_T345);\n  X_T671 = cond(X_T670, X_T345, X_T669);\n  X_T672 = sqrt(X_T671);\n  X_T673 = div(X_T668, X_T672);\n  X_T674 = add(X_T673, X_I_130);\n  X_T675 = cmp_lt(X_T674, X_T345);\n  X_T676 = cond(X_T675, X_T356, X_T674);\n  X_T292[n, x0, x1, c + m : 16, 19, 19, 728] = +(X_T676[n, -1 + k0 + x0, -1 + k1 + x1, c] * X_I_131[k0, k1, c, m]);\n  X_T291[n, x0, x1, co : 16, 19, 19, 728] = +(X_T292[n, k0 + x0, k1 + x1, ci] * X_I_132[k0, k1, ci, co]);\n  X_T681 = sub(X_T291, X_I_133);\n  X_T682 = mul(X_T681, X_I_134);\n  X_T683 = add(X_I_135, X_T348);\n  X_T684 = cmp_lt(X_T683, X_T345);\n  X_T685 = cond(X_T684, X_T345, X_T683);\n  X_T686 = sqrt(X_T685);\n  X_T687 = div(X_T682, X_T686);\n  X_T688 = add(X_T687, X_I_136);\n  X_T689 = add(X_T688, X_T646);\n  X_T690 = cmp_lt(X_T689, X_T345);\n  X_T691 = cond(X_T690, X_T356, X_T689);\n  X_T290[n, x0, x1, c + m : 16, 19, 19, 728] = +(X_T691[n, -1 + k0 + x0, -1 + k1 + x1, c] * X_I_137[k0, k1, c, m]);\n  X_T289[n, x0, x1, co : 16, 19, 19, 728] = +(X_T290[n, k0 + x0, k1 + x1, ci] * X_I_138[k0, k1, ci, co]);\n  X_T696 = sub(X_T289, X_I_139);\n  X_T697 = mul(X_T696, X_I_140);\n  X_T698 = add(X_I_141, X_T348);\n  X_T699 = cmp_lt(X_T698, X_T345);\n  X_T700 = cond(X_T699, X_T345, X_T698);\n  X_T701 = sqrt(X_T700);\n  X_T702 = div(X_T697, X_T701);\n  X_T703 = add(X_T702, X_I_142);\n  X_T704 = cmp_lt(X_T703, X_T345);\n  X_T705 = cond(X_T704, X_T356, X_T703);\n  X_T288[n, x0, x1, c + m : 16, 19, 19, 728] = +(X_T705[n, -1 + k0 + x0, -1 + k1 + x1, c] * X_I_143[k0, k1, c, m]);\n  X_T287[n, x0, x1, co : 16, 19, 19, 728] = +(X_T288[n, k0 + x0, k1 + x1, ci] * X_I_144[k0, k1, ci, co]);\n  X_T710 = sub(X_T287, X_I_145);\n  X_T711 = mul(X_T710, X_I_146);\n  X_T712 = add(X_I_147, X_T348);\n  X_T713 = cmp_lt(X_T712, X_T345);\n  X_T714 = cond(X_T713, X_T345, X_T712);\n  X_T715 = sqrt(X_T714);\n  X_T716 = div(X_T711, X_T715);\n  X_T717 = add(X_T716, X_I_148);\n  X_T718 = cmp_lt(X_T717, X_T345);\n  X_T719 = cond(X_T718, X_T356, X_T717);\n  X_T286[n, x0, x1, c + m : 16, 19, 19, 728] = +(X_T719[n, -1 + k0 + x0, -1 + k1 + x1, c] * X_I_149[k0, k1, c, m]);\n  X_T285[n, x0, x1, co : 16, 19, 19, 728] = +(X_T286[n, k0 + x0, k1 + x1, ci] * X_I_150[k0, k1, ci, co]);\n  X_T724 = sub(X_T285, X_I_151);\n  X_T725 = mul(X_T724, X_I_152);\n  X_T726 = add(X_I_153, X_T348);\n  X_T727 = cmp_lt(X_T726, X_T345);\n  X_T728 = cond(X_T727, X_T345, X_T726);\n  X_T729 = sqrt(X_T728);\n  X_T730 = div(X_T725, X_T729);\n  X_T731 = add(X_T730, X_I_154);\n  X_T732 = add(X_T731, X_T689);\n  X_T733 = cmp_lt(X_T732, X_T345);\n  X_T734 = cond(X_T733, X_T356, X_T732);\n  X_T284[n, x0, x1, c + m : 16, 19, 19, 728] = +(X_T734[n, -1 + k0 + x0, -1 + k1 + x1, c] * X_I_155[k0, k1, c, m]);\n  X_T283[n, x0, x1, co : 16, 19, 19, 728] = +(X_T284[n, k0 + x0, k1 + x1, ci] * X_I_156[k0, k1, ci, co]);\n  X_T739 = sub(X_T283, X_I_157);\n  X_T740 = mul(X_T739, X_I_158);\n  X_T741 = add(X_I_159, X_T348);\n  X_T742 = cmp_lt(X_T741, X_T345);\n  X_T743 = cond(X_T742, X_T345, X_T741);\n  X_T744 = sqrt(X_T743);\n  X_T745 = div(X_T740, X_T744);\n  X_T746 = add(X_T745, X_I_160);\n  X_T747 = cmp_lt(X_T746, X_T345);\n  X_T748 = cond(X_T747, X_T356, X_T746);\n  X_T282[n, x0, x1, c + m : 16, 19, 19, 728] = +(X_T748[n, -1 + k0 + x0, -1 + k1 + x1, c] * X_I_161[k0, k1, c, m]);\n  X_T281[n, x0, x1, co : 16, 19, 19, 728] = +(X_T282[n, k0 + x0, k1 + x1, ci] * X_I_162[k0, k1, ci, co]);\n  X_T753 = sub(X_T281, X_I_163);\n  X_T754 = mul(X_T753, X_I_164);\n  X_T755 = add(X_I_165, X_T348);\n  X_T756 = cmp_lt(X_T755, X_T345);\n  X_T757 = cond(X_T756, X_T345, X_T755);\n  X_T758 = sqrt(X_T757);\n  X_T759 = div(X_T754, X_T758);\n  X_T760 = add(X_T759, X_I_166);\n  X_T761 = cmp_lt(X_T760, X_T345);\n  X_T762 = cond(X_T761, X_T356, X_T760);\n  X_T280[n, x0, x1, c + m : 16, 19, 19, 728] = +(X_T762[n, -1 + k0 + x0, -1 + k1 + x1, c] * X_I_167[k0, k1, c, m]);\n  X_T279[n, x0, x1, co : 16, 19, 19, 728] = +(X_T280[n, k0 + x0, k1 + x1, ci] * X_I_168[k0, k1, ci, co]);\n  X_T767 = sub(X_T279, X_I_169);\n  X_T768 = mul(X_T767, X_I_170);\n  X_T769 = add(X_I_171, X_T348);\n  X_T770 = cmp_lt(X_T769, X_T345);\n  X_T771 = cond(X_T770, X_T345, X_T769);\n  X_T772 = sqrt(X_T771);\n  X_T773 = div(X_T768, X_T772);\n  X_T774 = add(X_T773, X_I_172);\n  X_T775 = add(X_T774, X_T732);\n  X_T776 = cmp_lt(X_T775, X_T345);\n  X_T777 = cond(X_T776, X_T356, X_T775);\n  X_T278[n, x0, x1, c + m : 16, 19, 19, 728] = +(X_T777[n, -1 + k0 + x0, -1 + k1 + x1, c] * X_I_173[k0, k1, c, m]);\n  X_T277[n, x0, x1, co : 16, 19, 19, 728] = +(X_T278[n, k0 + x0, k1 + x1, ci] * X_I_174[k0, k1, ci, co]);\n  X_T782 = sub(X_T277, X_I_175);\n  X_T783 = mul(X_T782, X_I_176);\n  X_T784 = add(X_I_177, X_T348);\n  X_T785 = cmp_lt(X_T784, X_T345);\n  X_T786 = cond(X_T785, X_T345, X_T784);\n  X_T787 = sqrt(X_T786);\n  X_T788 = div(X_T783, X_T787);\n  X_T789 = add(X_T788, X_I_178);\n  X_T790 = cmp_lt(X_T789, X_T345);\n  X_T791 = cond(X_T790, X_T356, X_T789);\n  X_T276[n, x0, x1, c + m : 16, 19, 19, 728] = +(X_T791[n, -1 + k0 + x0, -1 + k1 + x1, c] * X_I_179[k0, k1, c, m]);\n  X_T275[n, x0, x1, co : 16, 19, 19, 728] = +(X_T276[n, k0 + x0, k1 + x1, ci] * X_I_180[k0, k1, ci, co]);\n  X_T796 = sub(X_T275, X_I_181);\n  X_T797 = mul(X_T796, X_I_182);\n  X_T798 = add(X_I_183, X_T348);\n  X_T799 = cmp_lt(X_T798, X_T345);\n  X_T800 = cond(X_T799, X_T345, X_T798);\n  X_T801 = sqrt(X_T800);\n  X_T802 = div(X_T797, X_T801);\n  X_T803 = add(X_T802, X_I_184);\n  X_T804 = cmp_lt(X_T803, X_T345);\n  X_T805 = cond(X_T804, X_T356, X_T803);\n  X_T274[n, x0, x1, c + m : 16, 19, 19, 728] = +(X_T805[n, -1 + k0 + x0, -1 + k1 + x1, c] * X_I_185[k0, k1, c, m]);\n  X_T273[n, x0, x1, co : 16, 19, 19, 728] = +(X_T274[n, k0 + x0, k1 + x1, ci] * X_I_186[k0, k1, ci, co]);\n  X_T810 = sub(X_T273, X_I_187);\n  X_T811 = mul(X_T810, X_I_188);\n  X_T812 = add(X_I_189, X_T348);\n  X_T813 = cmp_lt(X_T812, X_T345);\n  X_T814 = cond(X_T813, X_T345, X_T812);\n  X_T815 = sqrt(X_T814);\n  X_T816 = div(X_T811, X_T815);\n  X_T817 = add(X_T816, X_I_190);\n  X_T818 = add(X_T817, X_T775);\n  X_T819 = cmp_lt(X_T818, X_T345);\n  X_T820 = cond(X_T819, X_T356, X_T818);\n  X_T272[n, x0, x1, c + m : 16, 19, 19, 728] = +(X_T820[n, -1 + k0 + x0, -1 + k1 + x1, c] * X_I_191[k0, k1, c, m]);\n  X_T271[n, x0, x1, co : 16, 19, 19, 728] = +(X_T272[n, k0 + x0, k1 + x1, ci] * X_I_192[k0, k1, ci, co]);\n  X_T825 = sub(X_T271, X_I_193);\n  X_T826 = mul(X_T825, X_I_194);\n  X_T827 = add(X_I_195, X_T348);\n  X_T828 = cmp_lt(X_T827, X_T345);\n  X_T829 = cond(X_T828, X_T345, X_T827);\n  X_T830 = sqrt(X_T829);\n  X_T831 = div(X_T826, X_T830);\n  X_T832 = add(X_T831, X_I_196);\n  X_T833 = cmp_lt(X_T832, X_T345);\n  X_T834 = cond(X_T833, X_T356, X_T832);\n  X_T270[n, x0, x1, c + m : 16, 19, 19, 728] = +(X_T834[n, -1 + k0 + x0, -1 + k1 + x1, c] * X_I_197[k0, k1, c, m]);\n  X_T269[n, x0, x1, co : 16, 19, 19, 728] = +(X_T270[n, k0 + x0, k1 + x1, ci] * X_I_198[k0, k1, ci, co]);\n  X_T839 = sub(X_T269, X_I_199);\n  X_T840 = mul(X_T839, X_I_200);\n  X_T841 = add(X_I_201, X_T348);\n  X_T842 = cmp_lt(X_T841, X_T345);\n  X_T843 = cond(X_T842, X_T345, X_T841);\n  X_T844 = sqrt(X_T843);\n  X_T845 = div(X_T840, X_T844);\n  X_T846 = add(X_T845, X_I_202);\n  X_T847 = cmp_lt(X_T846, X_T345);\n  X_T848 = cond(X_T847, X_T356, X_T846);\n  X_T268[n, x0, x1, c + m : 16, 19, 19, 728] = +(X_T848[n, -1 + k0 + x0, -1 + k1 + x1, c] * X_I_203[k0, k1, c, m]);\n  X_T267[n, x0, x1, co : 16, 19, 19, 728] = +(X_T268[n, k0 + x0, k1 + x1, ci] * X_I_204[k0, k1, ci, co]);\n  X_T853 = sub(X_T267, X_I_205);\n  X_T854 = mul(X_T853, X_I_206);\n  X_T855 = add(X_I_207, X_T348);\n  X_T856 = cmp_lt(X_T855, X_T345);\n  X_T857 = cond(X_T856, X_T345, X_T855);\n  X_T858 = sqrt(X_T857);\n  X_T859 = div(X_T854, X_T858);\n  X_T860 = add(X_T859, X_I_208);\n  X_T861 = add(X_T860, X_T818);\n  X_T862 = cmp_lt(X_T861, X_T345);\n  X_T863 = cond(X_T862, X_T356, X_T861);\n  X_T266[n, x0, x1, c + m : 16, 19, 19, 728] = +(X_T863[n, -1 + k0 + x0, -1 + k1 + x1, c] * X_I_209[k0, k1, c, m]);\n  X_T265[n, x0, x1, co : 16, 19, 19, 728] = +(X_T266[n, k0 + x0, k1 + x1, ci] * X_I_210[k0, k1, ci, co]);\n  X_T868 = sub(X_T265, X_I_211);\n  X_T869 = mul(X_T868, X_I_212);\n  X_T870 = add(X_I_213, X_T348);\n  X_T871 = cmp_lt(X_T870, X_T345);\n  X_T872 = cond(X_T871, X_T345, X_T870);\n  X_T873 = sqrt(X_T872);\n  X_T874 = div(X_T869, X_T873);\n  X_T875 = add(X_T874, X_I_214);\n  X_T876 = cmp_lt(X_T875, X_T345);\n  X_T877 = cond(X_T876, X_T356, X_T875);\n  X_T263[n, x0, x1, c + m : 16, 19, 19, 728] = +(X_T877[n, -1 + k0 + x0, -1 + k1 + x1, c] * X_I_215[k0, k1, c, m]);\n  X_T262[n, x0, x1, co : 16, 19, 19, 1024] = +(X_T263[n, k0 + x0, k1 + x1, ci] * X_I_216[k0, k1, ci, co]);\n  X_T882 = sub(X_T262, X_I_217);\n  X_T883 = mul(X_T882, X_I_218);\n  X_T884 = add(X_I_219, X_T348);\n  X_T885 = cmp_lt(X_T884, X_T345);\n  X_T886 = cond(X_T885, X_T345, X_T884);\n  X_T887 = sqrt(X_T886);\n  X_T888 = div(X_T883, X_T887);\n  X_T889 = add(X_T888, X_I_220);\n  X_T261[n, x0, x1, c : 16, 10, 10, 1024] = >(X_T889[n, -1 + k0 + 2*x0, -1 + k1 + 2*x1, c]), k0 < 3, k1 < 3;\n  X_T896[n, x0, x1, co : 16, 10, 10, 1024] = +(X_T861[n, k0 + 2*x0, k1 + 2*x1, ci] * X_I_221[k0, k1, ci, co]);\n  X_T903 = sub(X_T896, X_I_222);\n  X_T904 = mul(X_T903, X_I_223);\n  X_T905 = add(X_I_224, X_T348);\n  X_T906 = cmp_lt(X_T905, X_T345);\n  X_T907 = cond(X_T906, X_T345, X_T905);\n  X_T908 = sqrt(X_T907);\n  X_T909 = div(X_T904, X_T908);\n  X_T910 = add(X_T909, X_I_225);\n  X_T911 = add(X_T261, X_T910);\n  X_T259[n, x0, x1, c + m : 16, 10, 10, 1024] = +(X_T911[n, -1 + k0 + x0, -1 + k1 + x1, c] * X_I_226[k0, k1, c, m]);\n  X_T258[n, x0, x1, co : 16, 10, 10, 1536] = +(X_T259[n, k0 + x0, k1 + x1, ci] * X_I_227[k0, k1, ci, co]);\n  X_T916 = sub(X_T258, X_I_228);\n  X_T917 = mul(X_T916, X_I_229);\n  X_T918 = add(X_I_230, X_T348);\n  X_T919 = cmp_lt(X_T918, X_T345);\n  X_T920 = cond(X_T919, X_T345, X_T918);\n  X_T921 = sqrt(X_T920);\n  X_T922 = div(X_T917, X_T921);\n  X_T923 = add(X_T922, X_I_231);\n  X_T924 = cmp_lt(X_T923, X_T345);\n  X_T925 = cond(X_T924, X_T356, X_T923);\n  X_T256[n, x0, x1, c + m : 16, 10, 10, 1536] = +(X_T925[n, -1 + k0 + x0, -1 + k1 + x1, c] * X_I_232[k0, k1, c, m]);\n  X_T5[n, x0, x1, co : 16, 10, 10, 2048] = +(X_T256[n, k0 + x0, k1 + x1, ci] * X_I_233[k0, k1, ci, co]);\n  X_T930 = sub(X_T5, X_I_234);\n  X_T931 = mul(X_T930, X_I_235);\n  X_T932 = add(X_I_236, X_T348);\n  X_T933 = cmp_lt(X_T932, X_T345);\n  X_T934 = cond(X_T933, X_T345, X_T932);\n  X_T935 = sqrt(X_T934);\n  X_T936 = div(X_T931, X_T935);\n  X_T937 = add(X_T936, X_I_237);\n  X_T938 = cmp_lt(X_T937, X_T345);\n  X_T939 = cond(X_T938, X_T356, X_T937);\n  X_T3[x0, x3 : 16, 2048] = +(X_T939[x0, x1, x2, x3]);\n  X_T940 = mul(X_T255, X_T255);\n  X_T941 = div(X_T3, X_T940);\n  X_T0[x0, y1 : 16, 1000] = +(X_T941[x0, z] * X_I_238[z, y1]);\n  X_T942 = add(X_T0, X_I_239);\n  X_T1028[i : 16] = >(X_T942[i, j]);\n  X_T1029 = neg(X_T1028);\n  X_T1030[i, j : 16, 1000] = +(X_T942[i, j] + X_T1029[i]);\n  X_T1031 = exp(X_T1030);\n  X_T1032[i : 16] = +(X_T1031[i, j]);\n  X_T1033 = 1.0;\n  X_T1034 = div(X_T1033, X_T1032);\n  X_T943[i, j : 16, 1000] = +(X_T1031[i, j] * X_T1034[i]);\n}\n"
inputs {
  key: "X_I_0"
  value {
    type: FLOAT32
    dimensions {
      size: 6
      stride: 4
    }
    dimensions {
      size: 4
      stride: 1
    }
  }
}
inputs {
  key: "X_I_1"
  value {
    type: FLOAT32
    dimensions {
      size: 6
      stride: 6
    }
    dimensions {
      size: 6
      stride: 1
    }
  }
}
inputs {
  key: "X_I_10"
  value {
    type: FLOAT32
    dimensions {
      size: 64
      stride: 1
    }
  }
}
inputs {
  key: "X_I_100"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_101"
  value {
    type: FLOAT32
    dimensions {
      size: 3
      stride: 2184
    }
    dimensions {
      size: 3
      stride: 728
    }
    dimensions {
      size: 728
      stride: 1
    }
    dimensions {
      size: 1
      stride: 1
    }
  }
}
inputs {
  key: "X_I_102"
  value {
    type: FLOAT32
    dimensions {
      size: 1
      stride: 529984
    }
    dimensions {
      size: 1
      stride: 529984
    }
    dimensions {
      size: 728
      stride: 728
    }
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_103"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_104"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_105"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_106"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_107"
  value {
    type: FLOAT32
    dimensions {
      size: 3
      stride: 2184
    }
    dimensions {
      size: 3
      stride: 728
    }
    dimensions {
      size: 728
      stride: 1
    }
    dimensions {
      size: 1
      stride: 1
    }
  }
}
inputs {
  key: "X_I_108"
  value {
    type: FLOAT32
    dimensions {
      size: 1
      stride: 529984
    }
    dimensions {
      size: 1
      stride: 529984
    }
    dimensions {
      size: 728
      stride: 728
    }
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_109"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_11"
  value {
    type: FLOAT32
    dimensions {
      size: 64
      stride: 1
    }
  }
}
inputs {
  key: "X_I_110"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_111"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_112"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_113"
  value {
    type: FLOAT32
    dimensions {
      size: 3
      stride: 2184
    }
    dimensions {
      size: 3
      stride: 728
    }
    dimensions {
      size: 728
      stride: 1
    }
    dimensions {
      size: 1
      stride: 1
    }
  }
}
inputs {
  key: "X_I_114"
  value {
    type: FLOAT32
    dimensions {
      size: 1
      stride: 529984
    }
    dimensions {
      size: 1
      stride: 529984
    }
    dimensions {
      size: 728
      stride: 728
    }
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_115"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_116"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_117"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_118"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_119"
  value {
    type: FLOAT32
    dimensions {
      size: 3
      stride: 2184
    }
    dimensions {
      size: 3
      stride: 728
    }
    dimensions {
      size: 728
      stride: 1
    }
    dimensions {
      size: 1
      stride: 1
    }
  }
}
inputs {
  key: "X_I_12"
  value {
    type: FLOAT32
    dimensions {
      size: 64
      stride: 1
    }
  }
}
inputs {
  key: "X_I_120"
  value {
    type: FLOAT32
    dimensions {
      size: 1
      stride: 529984
    }
    dimensions {
      size: 1
      stride: 529984
    }
    dimensions {
      size: 728
      stride: 728
    }
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_121"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_122"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_123"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_124"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_125"
  value {
    type: FLOAT32
    dimensions {
      size: 3
      stride: 2184
    }
    dimensions {
      size: 3
      stride: 728
    }
    dimensions {
      size: 728
      stride: 1
    }
    dimensions {
      size: 1
      stride: 1
    }
  }
}
inputs {
  key: "X_I_126"
  value {
    type: FLOAT32
    dimensions {
      size: 1
      stride: 529984
    }
    dimensions {
      size: 1
      stride: 529984
    }
    dimensions {
      size: 728
      stride: 728
    }
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_127"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_128"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_129"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_13"
  value {
    type: FLOAT32
    dimensions {
      size: 64
      stride: 1
    }
  }
}
inputs {
  key: "X_I_130"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_131"
  value {
    type: FLOAT32
    dimensions {
      size: 3
      stride: 2184
    }
    dimensions {
      size: 3
      stride: 728
    }
    dimensions {
      size: 728
      stride: 1
    }
    dimensions {
      size: 1
      stride: 1
    }
  }
}
inputs {
  key: "X_I_132"
  value {
    type: FLOAT32
    dimensions {
      size: 1
      stride: 529984
    }
    dimensions {
      size: 1
      stride: 529984
    }
    dimensions {
      size: 728
      stride: 728
    }
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_133"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_134"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_135"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_136"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_137"
  value {
    type: FLOAT32
    dimensions {
      size: 3
      stride: 2184
    }
    dimensions {
      size: 3
      stride: 728
    }
    dimensions {
      size: 728
      stride: 1
    }
    dimensions {
      size: 1
      stride: 1
    }
  }
}
inputs {
  key: "X_I_138"
  value {
    type: FLOAT32
    dimensions {
      size: 1
      stride: 529984
    }
    dimensions {
      size: 1
      stride: 529984
    }
    dimensions {
      size: 728
      stride: 728
    }
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_139"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_14"
  value {
    type: FLOAT32
    dimensions {
      size: 3
      stride: 192
    }
    dimensions {
      size: 3
      stride: 64
    }
    dimensions {
      size: 64
      stride: 1
    }
    dimensions {
      size: 1
      stride: 1
    }
  }
}
inputs {
  key: "X_I_140"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_141"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_142"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_143"
  value {
    type: FLOAT32
    dimensions {
      size: 3
      stride: 2184
    }
    dimensions {
      size: 3
      stride: 728
    }
    dimensions {
      size: 728
      stride: 1
    }
    dimensions {
      size: 1
      stride: 1
    }
  }
}
inputs {
  key: "X_I_144"
  value {
    type: FLOAT32
    dimensions {
      size: 1
      stride: 529984
    }
    dimensions {
      size: 1
      stride: 529984
    }
    dimensions {
      size: 728
      stride: 728
    }
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_145"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_146"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_147"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_148"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_149"
  value {
    type: FLOAT32
    dimensions {
      size: 3
      stride: 2184
    }
    dimensions {
      size: 3
      stride: 728
    }
    dimensions {
      size: 728
      stride: 1
    }
    dimensions {
      size: 1
      stride: 1
    }
  }
}
inputs {
  key: "X_I_15"
  value {
    type: FLOAT32
    dimensions {
      size: 1
      stride: 8192
    }
    dimensions {
      size: 1
      stride: 8192
    }
    dimensions {
      size: 64
      stride: 128
    }
    dimensions {
      size: 128
      stride: 1
    }
  }
}
inputs {
  key: "X_I_150"
  value {
    type: FLOAT32
    dimensions {
      size: 1
      stride: 529984
    }
    dimensions {
      size: 1
      stride: 529984
    }
    dimensions {
      size: 728
      stride: 728
    }
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_151"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_152"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_153"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_154"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_155"
  value {
    type: FLOAT32
    dimensions {
      size: 3
      stride: 2184
    }
    dimensions {
      size: 3
      stride: 728
    }
    dimensions {
      size: 728
      stride: 1
    }
    dimensions {
      size: 1
      stride: 1
    }
  }
}
inputs {
  key: "X_I_156"
  value {
    type: FLOAT32
    dimensions {
      size: 1
      stride: 529984
    }
    dimensions {
      size: 1
      stride: 529984
    }
    dimensions {
      size: 728
      stride: 728
    }
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_157"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_158"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_159"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_16"
  value {
    type: FLOAT32
    dimensions {
      size: 128
      stride: 1
    }
  }
}
inputs {
  key: "X_I_160"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_161"
  value {
    type: FLOAT32
    dimensions {
      size: 3
      stride: 2184
    }
    dimensions {
      size: 3
      stride: 728
    }
    dimensions {
      size: 728
      stride: 1
    }
    dimensions {
      size: 1
      stride: 1
    }
  }
}
inputs {
  key: "X_I_162"
  value {
    type: FLOAT32
    dimensions {
      size: 1
      stride: 529984
    }
    dimensions {
      size: 1
      stride: 529984
    }
    dimensions {
      size: 728
      stride: 728
    }
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_163"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_164"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_165"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_166"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_167"
  value {
    type: FLOAT32
    dimensions {
      size: 3
      stride: 2184
    }
    dimensions {
      size: 3
      stride: 728
    }
    dimensions {
      size: 728
      stride: 1
    }
    dimensions {
      size: 1
      stride: 1
    }
  }
}
inputs {
  key: "X_I_168"
  value {
    type: FLOAT32
    dimensions {
      size: 1
      stride: 529984
    }
    dimensions {
      size: 1
      stride: 529984
    }
    dimensions {
      size: 728
      stride: 728
    }
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_169"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_17"
  value {
    type: FLOAT32
    dimensions {
      size: 128
      stride: 1
    }
  }
}
inputs {
  key: "X_I_170"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_171"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_172"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_173"
  value {
    type: FLOAT32
    dimensions {
      size: 3
      stride: 2184
    }
    dimensions {
      size: 3
      stride: 728
    }
    dimensions {
      size: 728
      stride: 1
    }
    dimensions {
      size: 1
      stride: 1
    }
  }
}
inputs {
  key: "X_I_174"
  value {
    type: FLOAT32
    dimensions {
      size: 1
      stride: 529984
    }
    dimensions {
      size: 1
      stride: 529984
    }
    dimensions {
      size: 728
      stride: 728
    }
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_175"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_176"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_177"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_178"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_179"
  value {
    type: FLOAT32
    dimensions {
      size: 3
      stride: 2184
    }
    dimensions {
      size: 3
      stride: 728
    }
    dimensions {
      size: 728
      stride: 1
    }
    dimensions {
      size: 1
      stride: 1
    }
  }
}
inputs {
  key: "X_I_18"
  value {
    type: FLOAT32
    dimensions {
      size: 128
      stride: 1
    }
  }
}
inputs {
  key: "X_I_180"
  value {
    type: FLOAT32
    dimensions {
      size: 1
      stride: 529984
    }
    dimensions {
      size: 1
      stride: 529984
    }
    dimensions {
      size: 728
      stride: 728
    }
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_181"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_182"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_183"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_184"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_185"
  value {
    type: FLOAT32
    dimensions {
      size: 3
      stride: 2184
    }
    dimensions {
      size: 3
      stride: 728
    }
    dimensions {
      size: 728
      stride: 1
    }
    dimensions {
      size: 1
      stride: 1
    }
  }
}
inputs {
  key: "X_I_186"
  value {
    type: FLOAT32
    dimensions {
      size: 1
      stride: 529984
    }
    dimensions {
      size: 1
      stride: 529984
    }
    dimensions {
      size: 728
      stride: 728
    }
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_187"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_188"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_189"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_19"
  value {
    type: FLOAT32
    dimensions {
      size: 128
      stride: 1
    }
  }
}
inputs {
  key: "X_I_190"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_191"
  value {
    type: FLOAT32
    dimensions {
      size: 3
      stride: 2184
    }
    dimensions {
      size: 3
      stride: 728
    }
    dimensions {
      size: 728
      stride: 1
    }
    dimensions {
      size: 1
      stride: 1
    }
  }
}
inputs {
  key: "X_I_192"
  value {
    type: FLOAT32
    dimensions {
      size: 1
      stride: 529984
    }
    dimensions {
      size: 1
      stride: 529984
    }
    dimensions {
      size: 728
      stride: 728
    }
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_193"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_194"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_195"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_196"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_197"
  value {
    type: FLOAT32
    dimensions {
      size: 3
      stride: 2184
    }
    dimensions {
      size: 3
      stride: 728
    }
    dimensions {
      size: 728
      stride: 1
    }
    dimensions {
      size: 1
      stride: 1
    }
  }
}
inputs {
  key: "X_I_198"
  value {
    type: FLOAT32
    dimensions {
      size: 1
      stride: 529984
    }
    dimensions {
      size: 1
      stride: 529984
    }
    dimensions {
      size: 728
      stride: 728
    }
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_199"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_2"
  value {
    type: FLOAT32
    dimensions {
      size: 16
      stride: 268203
    }
    dimensions {
      size: 299
      stride: 897
    }
    dimensions {
      size: 299
      stride: 3
    }
    dimensions {
      size: 3
      stride: 1
    }
  }
}
inputs {
  key: "X_I_20"
  value {
    type: FLOAT32
    dimensions {
      size: 3
      stride: 384
    }
    dimensions {
      size: 3
      stride: 128
    }
    dimensions {
      size: 128
      stride: 1
    }
    dimensions {
      size: 1
      stride: 1
    }
  }
}
inputs {
  key: "X_I_200"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_201"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_202"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_203"
  value {
    type: FLOAT32
    dimensions {
      size: 3
      stride: 2184
    }
    dimensions {
      size: 3
      stride: 728
    }
    dimensions {
      size: 728
      stride: 1
    }
    dimensions {
      size: 1
      stride: 1
    }
  }
}
inputs {
  key: "X_I_204"
  value {
    type: FLOAT32
    dimensions {
      size: 1
      stride: 529984
    }
    dimensions {
      size: 1
      stride: 529984
    }
    dimensions {
      size: 728
      stride: 728
    }
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_205"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_206"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_207"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_208"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_209"
  value {
    type: FLOAT32
    dimensions {
      size: 3
      stride: 2184
    }
    dimensions {
      size: 3
      stride: 728
    }
    dimensions {
      size: 728
      stride: 1
    }
    dimensions {
      size: 1
      stride: 1
    }
  }
}
inputs {
  key: "X_I_21"
  value {
    type: FLOAT32
    dimensions {
      size: 1
      stride: 16384
    }
    dimensions {
      size: 1
      stride: 16384
    }
    dimensions {
      size: 128
      stride: 128
    }
    dimensions {
      size: 128
      stride: 1
    }
  }
}
inputs {
  key: "X_I_210"
  value {
    type: FLOAT32
    dimensions {
      size: 1
      stride: 529984
    }
    dimensions {
      size: 1
      stride: 529984
    }
    dimensions {
      size: 728
      stride: 728
    }
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_211"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_212"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_213"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_214"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_215"
  value {
    type: FLOAT32
    dimensions {
      size: 3
      stride: 2184
    }
    dimensions {
      size: 3
      stride: 728
    }
    dimensions {
      size: 728
      stride: 1
    }
    dimensions {
      size: 1
      stride: 1
    }
  }
}
inputs {
  key: "X_I_216"
  value {
    type: FLOAT32
    dimensions {
      size: 1
      stride: 745472
    }
    dimensions {
      size: 1
      stride: 745472
    }
    dimensions {
      size: 728
      stride: 1024
    }
    dimensions {
      size: 1024
      stride: 1
    }
  }
}
inputs {
  key: "X_I_217"
  value {
    type: FLOAT32
    dimensions {
      size: 1024
      stride: 1
    }
  }
}
inputs {
  key: "X_I_218"
  value {
    type: FLOAT32
    dimensions {
      size: 1024
      stride: 1
    }
  }
}
inputs {
  key: "X_I_219"
  value {
    type: FLOAT32
    dimensions {
      size: 1024
      stride: 1
    }
  }
}
inputs {
  key: "X_I_22"
  value {
    type: FLOAT32
    dimensions {
      size: 128
      stride: 1
    }
  }
}
inputs {
  key: "X_I_220"
  value {
    type: FLOAT32
    dimensions {
      size: 1024
      stride: 1
    }
  }
}
inputs {
  key: "X_I_221"
  value {
    type: FLOAT32
    dimensions {
      size: 1
      stride: 745472
    }
    dimensions {
      size: 1
      stride: 745472
    }
    dimensions {
      size: 728
      stride: 1024
    }
    dimensions {
      size: 1024
      stride: 1
    }
  }
}
inputs {
  key: "X_I_222"
  value {
    type: FLOAT32
    dimensions {
      size: 1024
      stride: 1
    }
  }
}
inputs {
  key: "X_I_223"
  value {
    type: FLOAT32
    dimensions {
      size: 1024
      stride: 1
    }
  }
}
inputs {
  key: "X_I_224"
  value {
    type: FLOAT32
    dimensions {
      size: 1024
      stride: 1
    }
  }
}
inputs {
  key: "X_I_225"
  value {
    type: FLOAT32
    dimensions {
      size: 1024
      stride: 1
    }
  }
}
inputs {
  key: "X_I_226"
  value {
    type: FLOAT32
    dimensions {
      size: 3
      stride: 3072
    }
    dimensions {
      size: 3
      stride: 1024
    }
    dimensions {
      size: 1024
      stride: 1
    }
    dimensions {
      size: 1
      stride: 1
    }
  }
}
inputs {
  key: "X_I_227"
  value {
    type: FLOAT32
    dimensions {
      size: 1
      stride: 1572864
    }
    dimensions {
      size: 1
      stride: 1572864
    }
    dimensions {
      size: 1024
      stride: 1536
    }
    dimensions {
      size: 1536
      stride: 1
    }
  }
}
inputs {
  key: "X_I_228"
  value {
    type: FLOAT32
    dimensions {
      size: 1536
      stride: 1
    }
  }
}
inputs {
  key: "X_I_229"
  value {
    type: FLOAT32
    dimensions {
      size: 1536
      stride: 1
    }
  }
}
inputs {
  key: "X_I_23"
  value {
    type: FLOAT32
    dimensions {
      size: 128
      stride: 1
    }
  }
}
inputs {
  key: "X_I_230"
  value {
    type: FLOAT32
    dimensions {
      size: 1536
      stride: 1
    }
  }
}
inputs {
  key: "X_I_231"
  value {
    type: FLOAT32
    dimensions {
      size: 1536
      stride: 1
    }
  }
}
inputs {
  key: "X_I_232"
  value {
    type: FLOAT32
    dimensions {
      size: 3
      stride: 4608
    }
    dimensions {
      size: 3
      stride: 1536
    }
    dimensions {
      size: 1536
      stride: 1
    }
    dimensions {
      size: 1
      stride: 1
    }
  }
}
inputs {
  key: "X_I_233"
  value {
    type: FLOAT32
    dimensions {
      size: 1
      stride: 3145728
    }
    dimensions {
      size: 1
      stride: 3145728
    }
    dimensions {
      size: 1536
      stride: 2048
    }
    dimensions {
      size: 2048
      stride: 1
    }
  }
}
inputs {
  key: "X_I_234"
  value {
    type: FLOAT32
    dimensions {
      size: 2048
      stride: 1
    }
  }
}
inputs {
  key: "X_I_235"
  value {
    type: FLOAT32
    dimensions {
      size: 2048
      stride: 1
    }
  }
}
inputs {
  key: "X_I_236"
  value {
    type: FLOAT32
    dimensions {
      size: 2048
      stride: 1
    }
  }
}
inputs {
  key: "X_I_237"
  value {
    type: FLOAT32
    dimensions {
      size: 2048
      stride: 1
    }
  }
}
inputs {
  key: "X_I_238"
  value {
    type: FLOAT32
    dimensions {
      size: 2048
      stride: 1000
    }
    dimensions {
      size: 1000
      stride: 1
    }
  }
}
inputs {
  key: "X_I_239"
  value {
    type: FLOAT32
    dimensions {
      size: 1000
      stride: 1
    }
  }
}
inputs {
  key: "X_I_24"
  value {
    type: FLOAT32
    dimensions {
      size: 128
      stride: 1
    }
  }
}
inputs {
  key: "X_I_25"
  value {
    type: FLOAT32
    dimensions {
      size: 128
      stride: 1
    }
  }
}
inputs {
  key: "X_I_26"
  value {
    type: FLOAT32
    dimensions {
      size: 1
      stride: 8192
    }
    dimensions {
      size: 1
      stride: 8192
    }
    dimensions {
      size: 64
      stride: 128
    }
    dimensions {
      size: 128
      stride: 1
    }
  }
}
inputs {
  key: "X_I_27"
  value {
    type: FLOAT32
    dimensions {
      size: 128
      stride: 1
    }
  }
}
inputs {
  key: "X_I_28"
  value {
    type: FLOAT32
    dimensions {
      size: 128
      stride: 1
    }
  }
}
inputs {
  key: "X_I_29"
  value {
    type: FLOAT32
    dimensions {
      size: 128
      stride: 1
    }
  }
}
inputs {
  key: "X_I_3"
  value {
    type: FLOAT32
    dimensions {
      size: 3
      stride: 288
    }
    dimensions {
      size: 3
      stride: 96
    }
    dimensions {
      size: 3
      stride: 32
    }
    dimensions {
      size: 32
      stride: 1
    }
  }
}
inputs {
  key: "X_I_30"
  value {
    type: FLOAT32
    dimensions {
      size: 128
      stride: 1
    }
  }
}
inputs {
  key: "X_I_31"
  value {
    type: FLOAT32
    dimensions {
      size: 3
      stride: 384
    }
    dimensions {
      size: 3
      stride: 128
    }
    dimensions {
      size: 128
      stride: 1
    }
    dimensions {
      size: 1
      stride: 1
    }
  }
}
inputs {
  key: "X_I_32"
  value {
    type: FLOAT32
    dimensions {
      size: 1
      stride: 32768
    }
    dimensions {
      size: 1
      stride: 32768
    }
    dimensions {
      size: 128
      stride: 256
    }
    dimensions {
      size: 256
      stride: 1
    }
  }
}
inputs {
  key: "X_I_33"
  value {
    type: FLOAT32
    dimensions {
      size: 256
      stride: 1
    }
  }
}
inputs {
  key: "X_I_34"
  value {
    type: FLOAT32
    dimensions {
      size: 256
      stride: 1
    }
  }
}
inputs {
  key: "X_I_35"
  value {
    type: FLOAT32
    dimensions {
      size: 256
      stride: 1
    }
  }
}
inputs {
  key: "X_I_36"
  value {
    type: FLOAT32
    dimensions {
      size: 256
      stride: 1
    }
  }
}
inputs {
  key: "X_I_37"
  value {
    type: FLOAT32
    dimensions {
      size: 3
      stride: 768
    }
    dimensions {
      size: 3
      stride: 256
    }
    dimensions {
      size: 256
      stride: 1
    }
    dimensions {
      size: 1
      stride: 1
    }
  }
}
inputs {
  key: "X_I_38"
  value {
    type: FLOAT32
    dimensions {
      size: 1
      stride: 65536
    }
    dimensions {
      size: 1
      stride: 65536
    }
    dimensions {
      size: 256
      stride: 256
    }
    dimensions {
      size: 256
      stride: 1
    }
  }
}
inputs {
  key: "X_I_39"
  value {
    type: FLOAT32
    dimensions {
      size: 256
      stride: 1
    }
  }
}
inputs {
  key: "X_I_4"
  value {
    type: FLOAT32
    dimensions {
      size: 32
      stride: 1
    }
  }
}
inputs {
  key: "X_I_40"
  value {
    type: FLOAT32
    dimensions {
      size: 256
      stride: 1
    }
  }
}
inputs {
  key: "X_I_41"
  value {
    type: FLOAT32
    dimensions {
      size: 256
      stride: 1
    }
  }
}
inputs {
  key: "X_I_42"
  value {
    type: FLOAT32
    dimensions {
      size: 256
      stride: 1
    }
  }
}
inputs {
  key: "X_I_43"
  value {
    type: FLOAT32
    dimensions {
      size: 1
      stride: 32768
    }
    dimensions {
      size: 1
      stride: 32768
    }
    dimensions {
      size: 128
      stride: 256
    }
    dimensions {
      size: 256
      stride: 1
    }
  }
}
inputs {
  key: "X_I_44"
  value {
    type: FLOAT32
    dimensions {
      size: 256
      stride: 1
    }
  }
}
inputs {
  key: "X_I_45"
  value {
    type: FLOAT32
    dimensions {
      size: 256
      stride: 1
    }
  }
}
inputs {
  key: "X_I_46"
  value {
    type: FLOAT32
    dimensions {
      size: 256
      stride: 1
    }
  }
}
inputs {
  key: "X_I_47"
  value {
    type: FLOAT32
    dimensions {
      size: 256
      stride: 1
    }
  }
}
inputs {
  key: "X_I_48"
  value {
    type: FLOAT32
    dimensions {
      size: 3
      stride: 768
    }
    dimensions {
      size: 3
      stride: 256
    }
    dimensions {
      size: 256
      stride: 1
    }
    dimensions {
      size: 1
      stride: 1
    }
  }
}
inputs {
  key: "X_I_49"
  value {
    type: FLOAT32
    dimensions {
      size: 1
      stride: 186368
    }
    dimensions {
      size: 1
      stride: 186368
    }
    dimensions {
      size: 256
      stride: 728
    }
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_5"
  value {
    type: FLOAT32
    dimensions {
      size: 32
      stride: 1
    }
  }
}
inputs {
  key: "X_I_50"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_51"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_52"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_53"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_54"
  value {
    type: FLOAT32
    dimensions {
      size: 3
      stride: 2184
    }
    dimensions {
      size: 3
      stride: 728
    }
    dimensions {
      size: 728
      stride: 1
    }
    dimensions {
      size: 1
      stride: 1
    }
  }
}
inputs {
  key: "X_I_55"
  value {
    type: FLOAT32
    dimensions {
      size: 1
      stride: 529984
    }
    dimensions {
      size: 1
      stride: 529984
    }
    dimensions {
      size: 728
      stride: 728
    }
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_56"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_57"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_58"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_59"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_6"
  value {
    type: FLOAT32
    dimensions {
      size: 32
      stride: 1
    }
  }
}
inputs {
  key: "X_I_60"
  value {
    type: FLOAT32
    dimensions {
      size: 1
      stride: 186368
    }
    dimensions {
      size: 1
      stride: 186368
    }
    dimensions {
      size: 256
      stride: 728
    }
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_61"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_62"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_63"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_64"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_65"
  value {
    type: FLOAT32
    dimensions {
      size: 3
      stride: 2184
    }
    dimensions {
      size: 3
      stride: 728
    }
    dimensions {
      size: 728
      stride: 1
    }
    dimensions {
      size: 1
      stride: 1
    }
  }
}
inputs {
  key: "X_I_66"
  value {
    type: FLOAT32
    dimensions {
      size: 1
      stride: 529984
    }
    dimensions {
      size: 1
      stride: 529984
    }
    dimensions {
      size: 728
      stride: 728
    }
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_67"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_68"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_69"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_7"
  value {
    type: FLOAT32
    dimensions {
      size: 32
      stride: 1
    }
  }
}
inputs {
  key: "X_I_70"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_71"
  value {
    type: FLOAT32
    dimensions {
      size: 3
      stride: 2184
    }
    dimensions {
      size: 3
      stride: 728
    }
    dimensions {
      size: 728
      stride: 1
    }
    dimensions {
      size: 1
      stride: 1
    }
  }
}
inputs {
  key: "X_I_72"
  value {
    type: FLOAT32
    dimensions {
      size: 1
      stride: 529984
    }
    dimensions {
      size: 1
      stride: 529984
    }
    dimensions {
      size: 728
      stride: 728
    }
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_73"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_74"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_75"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_76"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_77"
  value {
    type: FLOAT32
    dimensions {
      size: 3
      stride: 2184
    }
    dimensions {
      size: 3
      stride: 728
    }
    dimensions {
      size: 728
      stride: 1
    }
    dimensions {
      size: 1
      stride: 1
    }
  }
}
inputs {
  key: "X_I_78"
  value {
    type: FLOAT32
    dimensions {
      size: 1
      stride: 529984
    }
    dimensions {
      size: 1
      stride: 529984
    }
    dimensions {
      size: 728
      stride: 728
    }
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_79"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_8"
  value {
    type: FLOAT32
    dimensions {
      size: 6
      stride: 3
    }
    dimensions {
      size: 3
      stride: 1
    }
  }
}
inputs {
  key: "X_I_80"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_81"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_82"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_83"
  value {
    type: FLOAT32
    dimensions {
      size: 3
      stride: 2184
    }
    dimensions {
      size: 3
      stride: 728
    }
    dimensions {
      size: 728
      stride: 1
    }
    dimensions {
      size: 1
      stride: 1
    }
  }
}
inputs {
  key: "X_I_84"
  value {
    type: FLOAT32
    dimensions {
      size: 1
      stride: 529984
    }
    dimensions {
      size: 1
      stride: 529984
    }
    dimensions {
      size: 728
      stride: 728
    }
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_85"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_86"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_87"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_88"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_89"
  value {
    type: FLOAT32
    dimensions {
      size: 3
      stride: 2184
    }
    dimensions {
      size: 3
      stride: 728
    }
    dimensions {
      size: 728
      stride: 1
    }
    dimensions {
      size: 1
      stride: 1
    }
  }
}
inputs {
  key: "X_I_9"
  value {
    type: FLOAT32
    dimensions {
      size: 3
      stride: 6144
    }
    dimensions {
      size: 3
      stride: 2048
    }
    dimensions {
      size: 32
      stride: 64
    }
    dimensions {
      size: 64
      stride: 1
    }
  }
}
inputs {
  key: "X_I_90"
  value {
    type: FLOAT32
    dimensions {
      size: 1
      stride: 529984
    }
    dimensions {
      size: 1
      stride: 529984
    }
    dimensions {
      size: 728
      stride: 728
    }
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_91"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_92"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_93"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_94"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_95"
  value {
    type: FLOAT32
    dimensions {
      size: 3
      stride: 2184
    }
    dimensions {
      size: 3
      stride: 728
    }
    dimensions {
      size: 728
      stride: 1
    }
    dimensions {
      size: 1
      stride: 1
    }
  }
}
inputs {
  key: "X_I_96"
  value {
    type: FLOAT32
    dimensions {
      size: 1
      stride: 529984
    }
    dimensions {
      size: 1
      stride: 529984
    }
    dimensions {
      size: 728
      stride: 728
    }
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_97"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_98"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
inputs {
  key: "X_I_99"
  value {
    type: FLOAT32
    dimensions {
      size: 728
      stride: 1
    }
  }
}
outputs {
  key: "X_T943"
  value {
    type: FLOAT32
    dimensions {
      size: 16
      stride: 1000
    }
    dimensions {
      size: 1000
      stride: 1
    }
  }
}
