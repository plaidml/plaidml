id: "prng"
code: "function (X_I_0[X_I_0_0, X_I_0_1]) -> (X_T2, X_T7) {X_T0 = 32; X_T1 = prng_step(X_I_0, X_I_0_0, X_I_0_0, X_I_0_0, X_T0); X_T2 = prng_state(X_T1); X_T3 = -0.13801311186847084; X_T4 = 0.2760262237369417; X_T5 = prng_value(X_T1); X_T6 = mul(X_T4, X_T5); X_T7 = add(X_T3, X_T6);}"
inputs {
  key:"X_I_0"
  value {
    shape {
      type: UINT32
      dims: { size:3 stride: 2048 }
      dims: { size:2048 stride: 1 }
    }
  }
}
outputs {
  key:"X_T2"
  value {
    shape {
      type: UINT32
      dims: { size:3 stride: 2048 }
      dims: { size:2048 stride: 1 }
    }
  }
}
outputs {
  key:"X_T7"
  value {
    shape {
      type: FLOAT32
      dims: { size:3 stride: 288 }
      dims: { size:3 stride: 96 }
      dims: { size:3 stride: 32 }
      dims: { size:32 stride: 1 }
    }
  }
}
