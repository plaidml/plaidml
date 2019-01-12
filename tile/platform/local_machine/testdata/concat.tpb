id: "concat"
code: "function (X_I_0[X_I_0_0], X_I_1[X_I_1_0], X_I_2[X_I_2_0]) -> (X_T5) { X_T0[a : 512] = =(X_I_0[a]); X_T2[128 + a : 512] = =(X_I_1[a]); X_T3 = add(X_T0, X_T2); X_T4[256 + a : 512] = =(X_I_2[a]); X_T5 = add(X_T3, X_T4);}"
inputs {
  key:"X_I_0"
  value {
    shape {
      type: FLOAT32
      dims: { size:128 stride: 1 }
    }
  }
}
inputs {
  key:"X_I_1"
  value {
    shape {
      type: FLOAT32
      dims: { size:128 stride: 1 }
    }
  }
}
inputs {
  key:"X_I_2"
  value {
    shape {
      type: FLOAT32
      dims: { size:256 stride: 1 }
    }
  }
}
outputs {
  key:"X_T5"
  value {
    shape {
      type: FLOAT32
      dims: { size:512 stride: 1 }
    }
  }
}
