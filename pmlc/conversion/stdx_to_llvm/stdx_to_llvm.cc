// namespace {

// /// A pass converting MLIR operations into the LLVM IR dialect.
// struct LLVMLoweringPass : public ModulePass<LLVMLoweringPass> {
//   // By default, the patterns are those converting Standard operations to the
//   // LLVMIR dialect.
//   explicit LLVMLoweringPass(bool useAlloca = false,
//                             LLVMPatternListFiller patternListFiller = populateStdToLLVMConversionPatterns,
//                             LLVMTypeConverterMaker converterBuilder = makeStandardToLLVMTypeConverter)
//       : patternListFiller(patternListFiller), typeConverterMaker(converterBuilder) {}

//   // Run the dialect converter on the module.
//   void runOnModule() override {
//     if (!typeConverterMaker || !patternListFiller) return signalPassFailure();

//     ModuleOp m = getModule();
//     LLVM::ensureDistinctSuccessors(m);
//     std::unique_ptr<LLVMTypeConverter> typeConverter = typeConverterMaker(&getContext());
//     if (!typeConverter) return signalPassFailure();

//     OwningRewritePatternList patterns;
//     populateLoopToStdConversionPatterns(patterns, m.getContext());
//     patternListFiller(*typeConverter, patterns);

//     ConversionTarget target(getContext());
//     target.addLegalDialect<LLVM::LLVMDialect>();
//     if (failed(applyPartialConversion(m, target, patterns, &*typeConverter))) signalPassFailure();
//   }

//   // Callback for creating a list of patterns.  It is called every time in
//   // runOnModule since applyPartialConversion consumes the list.
//   LLVMPatternListFiller patternListFiller;

//   // Callback for creating an instance of type converter.  The converter
//   // constructor needs an MLIRContext, which is not available until runOnModule.
//   LLVMTypeConverterMaker typeConverterMaker;
// };

// }  // end namespace

// static PassRegistration<LLVMLoweringPass> pass(  //
//     "convert-std-to-llvm",
//     "Convert scalar and vector operations from the "
//     "Standard to the LLVM dialect",
//     [] {
//       return std::make_unique<LLVMLoweringPass>(clUseAlloca.getValue(), populateStdToLLVMConversionPatterns,
//                                                 makeStandardToLLVMTypeConverter);
//     });
