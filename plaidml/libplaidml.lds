/* Copyright 2019, Intel Corporation
 *
 * This is the linker script used to produce the shared libplaidml library on
 * systems that use the GNU linker.
 */

VERSION {
	VERS_1.1 {
		global:
      /* Export functionality provided by libplaidml. */
			plaidml_*;
			tile_*;
			vai_*;

      /* Export symbols for basic dynamic library functionality. */
      __bss_start;
      _edata;
      _end;
      _fini;
      _init;

    /* Hide all other symbols.
     * This reduces conflicts between versions of libraries used by libplaidml
     * and other libraries included in the process, e.g. protocol buffers and
     * LLVM. */
	  local: *;
	};
};
