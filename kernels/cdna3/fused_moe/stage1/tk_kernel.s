
# __CLANG_OFFLOAD_BUNDLE____START__ hip-amdgcn-amd-amdhsa--gfx942
	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.protected	_Z6kernel18moe_stage1_globals ; -- Begin function _Z6kernel18moe_stage1_globals
	.globl	_Z6kernel18moe_stage1_globals
	.p2align	8
	.type	_Z6kernel18moe_stage1_globals,@function
_Z6kernel18moe_stage1_globals:          ; @_Z6kernel18moe_stage1_globals
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_dword s3, s[0:1], 0x108
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s33, s3, 3
	s_cmp_ge_i32 s2, s33
	s_cbranch_scc1 .LBB0_43
; %bb.1:
	s_mov_b64 s[6:7], src_shared_base
	s_cmp_lg_u32 0, -1
	s_cselect_b32 s6, s7, 0
	s_cselect_b32 s7, 0, 0
	s_and_b32 s20, s7, 15
	s_and_b32 s8, s7, -16
	s_add_u32 s8, s8, 16
	s_mov_b32 s21, 0
	s_addc_u32 s9, s6, 0
	s_cmp_eq_u64 s[20:21], 0
	s_cselect_b32 s10, s7, s8
	s_cselect_b32 s6, s6, s9
	s_add_u32 s7, s10, 0x2000
	s_addc_u32 s6, s6, 0
	s_and_b32 s20, s7, 15
	s_and_b32 s8, s7, -16
	s_add_u32 s8, s8, 16
	s_addc_u32 s9, s6, 0
	s_cmp_eq_u64 s[20:21], 0
	s_cselect_b32 s11, s7, s8
	s_cselect_b32 s6, s6, s9
	s_add_u32 s7, s11, 0x8000
	s_addc_u32 s6, s6, 0
	s_and_b32 s20, s7, 15
	s_and_b32 s8, s7, -16
	s_add_u32 s8, s8, 16
	s_addc_u32 s9, s6, 0
	s_cmp_eq_u64 s[20:21], 0
	s_cselect_b32 s24, s7, s8
	s_cselect_b32 s25, s6, s9
	s_add_u32 s6, s24, 0x80
	s_addc_u32 s7, s25, 0
	s_and_b32 s20, s6, 15
	s_and_b32 s8, s6, -16
	s_add_u32 s8, s8, 16
	s_addc_u32 s9, s7, 0
	s_cmp_eq_u64 s[20:21], 0
	s_cselect_b32 s6, s6, s8
	s_cselect_b32 s7, s7, s9
	s_add_u32 s8, s6, 0x100
	s_addc_u32 s9, s7, 0
	s_and_b32 s20, s8, 15
	s_and_b32 s12, s8, -16
	s_add_u32 s12, s12, 16
	s_addc_u32 s13, s9, 0
	s_cmp_eq_u64 s[20:21], 0
	v_lshlrev_b32_e32 v2, 4, v0
	s_cselect_b32 s8, s8, s12
	v_and_b32_e32 v24, 0xf0, v2
	v_lshlrev_b32_e32 v2, 9, v0
	s_movk_i32 s12, 0x80
	v_subrev_co_u32_e32 v4, vcc, s12, v24
	v_and_b32_e32 v7, 0x1000, v2
	v_lshrrev_b32_e32 v2, 4, v0
	v_or_b32_e32 v8, 8, v24
	v_add_u32_e32 v9, 0xffffff88, v24
	v_cndmask_b32_e32 v5, v4, v24, vcc
	v_cndmask_b32_e32 v9, v9, v8, vcc
	v_lshl_add_u32 v8, v2, 7, s10
	v_add_u32_e32 v10, v8, v5
	v_lshrrev_b32_e32 v12, 4, v10
	v_add_u32_e32 v11, v10, v7
	v_and_b32_e32 v12, 0x78, v12
	v_add_u32_e32 v8, v8, v9
	v_add_u32_e32 v10, 0x2000, v10
	v_xor_b32_e32 v25, v12, v11
	v_add_u32_e32 v12, v8, v7
	v_lshrrev_b32_e32 v13, 4, v8
	v_lshrrev_b32_e32 v10, 4, v10
	v_add_u32_e32 v8, 0x2000, v8
	v_or_b32_e32 v4, 32, v2
	v_and_b32_e32 v13, 0x78, v13
	v_add_u32_e32 v11, 0x2000, v11
	v_and_b32_e32 v10, 0x78, v10
	v_lshrrev_b32_e32 v8, 4, v8
	v_or_b32_e32 v6, 0x60, v2
	v_xor_b32_e32 v29, v13, v12
	v_lshl_add_u32 v13, v4, 7, s10
	v_xor_b32_e32 v62, v10, v11
	v_add_u32_e32 v10, 0x2000, v12
	v_and_b32_e32 v8, 0x78, v8
	v_add_u32_e32 v14, v13, v5
	v_xor_b32_e32 v63, v8, v10
	v_lshl_add_u32 v8, v6, 7, s10
	v_add_u32_e32 v15, v14, v7
	v_lshrrev_b32_e32 v14, 4, v14
	v_add_u32_e32 v10, v8, v5
	v_and_b32_e32 v14, 0x78, v14
	v_add_u32_e32 v13, v13, v9
	v_add_u32_e32 v11, v10, v7
	v_lshrrev_b32_e32 v10, 4, v10
	v_xor_b32_e32 v60, v14, v15
	v_add_u32_e32 v14, v13, v7
	v_lshrrev_b32_e32 v13, 4, v13
	v_and_b32_e32 v10, 0x78, v10
	v_add_u32_e32 v8, v8, v9
	v_and_b32_e32 v13, 0x78, v13
	v_xor_b32_e32 v64, v10, v11
	v_add_u32_e32 v10, v8, v7
	v_lshrrev_b32_e32 v8, 4, v8
	v_lshlrev_b32_e32 v12, 3, v0
	v_xor_b32_e32 v61, v13, v14
	v_and_b32_e32 v8, 0x78, v8
	v_and_b32_e32 v14, 0xf80, v12
	v_xor_b32_e32 v65, v8, v10
	v_add_u32_e32 v10, s11, v14
	v_lshlrev_b32_e32 v8, 11, v0
	v_add_u32_e32 v11, v10, v5
	v_and_b32_e32 v8, 0x4000, v8
	v_lshrrev_b32_e32 v15, 4, v11
	v_add_u32_e32 v13, v11, v8
	v_and_b32_e32 v15, 0x78, v15
	v_add_u32_e32 v10, v10, v9
	v_xor_b32_e32 v66, v15, v13
	v_lshrrev_b32_e32 v15, 4, v10
	v_add_u32_e32 v8, v10, v8
	v_and_b32_e32 v15, 0x78, v15
	v_xor_b32_e32 v67, v15, v8
	v_add_u32_e32 v15, 0x1000, v11
	v_lshrrev_b32_e32 v15, 4, v15
	v_add_u32_e32 v16, 0x1000, v13
	v_and_b32_e32 v15, 0x78, v15
	v_xor_b32_e32 v68, v15, v16
	v_add_u32_e32 v15, 0x1000, v10
	v_lshrrev_b32_e32 v15, 4, v15
	v_add_u32_e32 v16, 0x1000, v8
	v_and_b32_e32 v15, 0x78, v15
	v_xor_b32_e32 v69, v15, v16
	v_add_u32_e32 v15, 0x2000, v11
	v_lshrrev_b32_e32 v15, 4, v15
	v_add_u32_e32 v16, 0x2000, v13
	v_and_b32_e32 v15, 0x78, v15
	v_xor_b32_e32 v70, v15, v16
	v_add_u32_e32 v15, 0x2000, v10
	v_add_u32_e32 v10, 0x3000, v10
	v_lshrrev_b32_e32 v15, 4, v15
	v_lshrrev_b32_e32 v10, 4, v10
	v_add_u32_e32 v16, 0x2000, v8
	v_and_b32_e32 v15, 0x78, v15
	v_add_u32_e32 v8, 0x3000, v8
	v_and_b32_e32 v10, 0x78, v10
	v_and_b32_e32 v28, 15, v0
	v_bfe_u32 v3, v0, 6, 2
	v_xor_b32_e32 v71, v15, v16
	v_xor_b32_e32 v73, v10, v8
	v_lshrrev_b32_e32 v8, 1, v0
	v_lshlrev_b32_e32 v16, 7, v28
	v_and_b32_e32 v15, 24, v8
	v_lshl_or_b32 v8, v3, 11, v16
	v_add_u32_e32 v18, s11, v8
	s_load_dwordx2 s[4:5], s[0:1], 0x78
	v_add_u32_e32 v19, v18, v15
	v_lshrrev_b32_e32 v8, 4, v19
	v_and_b32_e32 v20, 0x78, v8
	v_lshlrev_b32_e32 v8, 2, v0
	v_add_u32_e32 v11, 0x3000, v11
	v_and_b32_e32 v21, 0xfc, v8
	s_cselect_b32 s9, s9, s13
	v_lshrrev_b32_e32 v1, 8, v0
	v_mov_b32_e32 v27, 0
	v_lshrrev_b32_e32 v11, 4, v11
	v_lshlrev_b32_e32 v26, 2, v21
	v_add_u32_e32 v13, 0x3000, v13
	v_and_b32_e32 v11, 0x78, v11
	v_lshl_add_u64 v[30:31], s[6:7], 0, v[26:27]
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[32:33], s[4:5], 0, v[26:27]
	v_lshl_add_u64 v[34:35], s[8:9], 0, v[26:27]
	v_lshlrev_b32_e32 v26, 6, v1
	v_and_b32_e32 v22, 63, v0
	v_xor_b32_e32 v72, v11, v13
	v_lshl_add_u64 v[10:11], s[24:25], 0, v[26:27]
	v_lshlrev_b32_e32 v26, 3, v22
	v_and_b32_e32 v23, 56, v12
	v_lshrrev_b32_e32 v12, 3, v0
	v_lshl_add_u64 v[36:37], v[10:11], 0, v[26:27]
	v_lshlrev_b32_e32 v26, 6, v3
	v_and_or_b32 v12, v12, 7, v23
	v_lshl_add_u64 v[10:11], s[6:7], 0, v[26:27]
	v_lshlrev_b32_e32 v12, 2, v12
	v_mov_b32_e32 v13, v27
	v_or_b32_e32 v17, 32, v15
	v_lshl_add_u64 v[38:39], v[10:11], 0, v[12:13]
	v_lshl_add_u64 v[10:11], s[8:9], 0, v[26:27]
	v_lshl_add_u64 v[40:41], v[10:11], 0, v[12:13]
	v_add_u32_e32 v10, v18, v17
	v_add_u32_e32 v11, 0x4000, v19
	v_xor_b32_e32 v75, v20, v11
	v_lshrrev_b32_e32 v11, 4, v10
	v_and_b32_e32 v11, 0x78, v11
	v_xor_b32_e32 v76, v11, v10
	v_add_u32_e32 v10, 0x4000, v10
	v_xor_b32_e32 v77, v11, v10
	v_or_b32_e32 v10, 64, v15
	v_add_u32_e32 v11, v18, v10
	v_lshrrev_b32_e32 v12, 4, v11
	v_and_b32_e32 v12, 0x78, v12
	v_xor_b32_e32 v78, v12, v11
	v_add_u32_e32 v11, 0x4000, v11
	v_xor_b32_e32 v79, v12, v11
	v_or_b32_e32 v11, 0x60, v15
	v_add_u32_e32 v12, v18, v11
	v_lshrrev_b32_e32 v13, 4, v12
	v_and_b32_e32 v13, 0x78, v13
	v_xor_b32_e32 v80, v13, v12
	v_add_u32_e32 v12, 0x4000, v12
	v_xor_b32_e32 v81, v13, v12
	v_lshl_or_b32 v12, v1, 11, v16
	v_add_u32_e32 v12, s10, v12
	v_add_u32_e32 v13, v12, v15
	v_lshrrev_b32_e32 v16, 4, v13
	v_and_b32_e32 v16, 0x78, v16
	v_xor_b32_e32 v82, v16, v13
	v_add_u32_e32 v13, 0x1000, v13
	v_xor_b32_e32 v83, v16, v13
	v_add_u32_e32 v13, v12, v17
	v_lshrrev_b32_e32 v16, 4, v13
	v_and_b32_e32 v16, 0x78, v16
	v_xor_b32_e32 v84, v16, v13
	v_add_u32_e32 v13, 0x1000, v13
	v_xor_b32_e32 v85, v16, v13
	v_add_u32_e32 v13, v12, v10
	v_lshrrev_b32_e32 v16, 4, v13
	v_and_b32_e32 v16, 0x78, v16
	v_xor_b32_e32 v86, v16, v13
	v_add_u32_e32 v13, 0x1000, v13
	v_add_u32_e32 v12, v12, v11
	v_xor_b32_e32 v87, v16, v13
	v_lshrrev_b32_e32 v13, 4, v12
	v_and_b32_e32 v13, 0x78, v13
	v_xor_b32_e32 v88, v13, v12
	v_add_u32_e32 v12, 0x1000, v12
	v_xor_b32_e32 v89, v13, v12
	v_add_u32_e32 v12, 0x2000, v18
	v_add_u32_e32 v13, v12, v15
	v_lshrrev_b32_e32 v16, 4, v13
	v_and_b32_e32 v16, 0x78, v16
	v_xor_b32_e32 v90, v16, v13
	v_add_u32_e32 v13, 0x4000, v13
	v_xor_b32_e32 v91, v16, v13
	v_add_u32_e32 v13, v12, v17
	v_lshrrev_b32_e32 v16, 4, v13
	v_and_b32_e32 v16, 0x78, v16
	v_xor_b32_e32 v92, v16, v13
	v_add_u32_e32 v13, 0x4000, v13
	v_xor_b32_e32 v93, v16, v13
	v_add_u32_e32 v13, v12, v10
	v_lshrrev_b32_e32 v16, 4, v13
	v_and_b32_e32 v16, 0x78, v16
	v_xor_b32_e32 v94, v16, v13
	v_add_u32_e32 v13, 0x4000, v13
	v_add_u32_e32 v12, v12, v11
	v_xor_b32_e32 v95, v16, v13
	v_lshrrev_b32_e32 v13, 4, v12
	v_and_b32_e32 v13, 0x78, v13
	v_xor_b32_e32 v96, v13, v12
	v_add_u32_e32 v12, 0x4000, v12
	v_xor_b32_e32 v97, v13, v12
	v_add_u32_e32 v12, s10, v14
	v_add_u32_e32 v9, v12, v9
	v_add_u32_e32 v5, v12, v5
	v_lshrrev_b32_e32 v13, 4, v9
	v_add_u32_e32 v9, v9, v7
	v_add_u32_e32 v7, v5, v7
	v_lshrrev_b32_e32 v5, 4, v5
	v_lshlrev_b32_e32 v101, 4, v3
	v_and_b32_e32 v5, 0x78, v5
	v_lshlrev_b32_e32 v100, 4, v1
	v_or_b32_e32 v3, v101, v28
	v_and_b32_e32 v13, 0x78, v13
	v_xor_b32_e32 v99, v5, v7
	v_or_b32_e32 v5, v100, v28
	v_lshl_add_u32 v3, v3, 7, s11
	v_xor_b32_e32 v98, v13, v9
	v_lshl_add_u32 v5, v5, 7, s10
	v_add_u32_e32 v13, v3, v15
	v_add_u32_e32 v7, v5, v15
	v_add_u32_e32 v9, v5, v17
	v_add_u32_e32 v12, v5, v10
	v_add_u32_e32 v5, v5, v11
	v_add_u32_e32 v14, v3, v17
	v_add_u32_e32 v10, v3, v10
	v_add_u32_e32 v3, v3, v11
	v_lshrrev_b32_e32 v11, 4, v13
	v_and_b32_e32 v11, 0x78, v11
	v_xor_b32_e32 v102, v11, v13
	v_add_u32_e32 v13, 0x4000, v13
	v_xor_b32_e32 v103, v11, v13
	v_lshrrev_b32_e32 v11, 4, v14
	s_load_dwordx4 s[36:39], s[0:1], 0x60
	s_load_dwordx2 s[40:41], s[0:1], 0x0
	v_and_b32_e32 v11, 0x78, v11
	v_add_u32_e32 v13, 0x4000, v14
	v_xor_b32_e32 v104, v11, v14
	v_xor_b32_e32 v105, v11, v13
	v_lshrrev_b32_e32 v11, 4, v10
	v_and_b32_e32 v11, 0x78, v11
	v_xor_b32_e32 v106, v11, v10
	v_add_u32_e32 v10, 0x4000, v10
	v_xor_b32_e32 v107, v11, v10
	v_lshrrev_b32_e32 v10, 4, v3
	s_load_dwordx4 s[16:19], s[0:1], 0x10
	s_load_dword s4, s[0:1], 0x38
	v_and_b32_e32 v10, 0x78, v10
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s22, s38, 7
	v_xor_b32_e32 v108, v10, v3
	v_add_u32_e32 v3, 0x4000, v3
	s_and_b32 s5, s38, 0x3fff
	v_xor_b32_e32 v109, v10, v3
	v_lshrrev_b32_e32 v3, 4, v7
	s_cselect_b64 s[34:35], 0, -1
	s_lshl_b32 s20, s38, 16
	v_and_b32_e32 v3, 0x78, v3
	s_and_b32 s5, s18, 0x3fff
	v_xor_b32_e32 v110, v3, v7
	v_add_u32_e32 v7, 0x1000, v7
	s_cselect_b64 s[42:43], 0, -1
	s_lshl_b32 s23, s18, 16
	s_lshl_b32 s26, s4, 2
	v_xor_b32_e32 v111, v3, v7
	v_lshrrev_b32_e32 v3, 4, v9
	s_add_u32 s48, s0, 0x118
	v_and_b32_e32 v3, 0x78, v3
	s_addc_u32 s49, s1, 0
	s_abs_i32 s17, s3
	v_xor_b32_e32 v112, v3, v9
	v_add_u32_e32 v7, 0x1000, v9
	v_cvt_f32_u32_e32 v9, s17
	v_xor_b32_e32 v113, v3, v7
	v_lshrrev_b32_e32 v3, 4, v12
	v_and_b32_e32 v3, 0x78, v3
	v_add_u32_e32 v7, 0x1000, v12
	v_xor_b32_e32 v115, v3, v7
	v_rcp_f32_e32 v7, v9
	v_xor_b32_e32 v114, v3, v12
	v_lshrrev_b32_e32 v3, 4, v5
	v_and_b32_e32 v3, 0x78, v3
	v_xor_b32_e32 v116, v3, v5
	v_add_u32_e32 v5, 0x1000, v5
	v_xor_b32_e32 v117, v3, v5
	v_mul_f32_e32 v3, 0x4f7ffffe, v7
	v_cvt_u32_f32_e32 v3, v3
	v_cmp_eq_u32_e64 s[6:7], 1, v1
	v_cmp_eq_u32_e64 s[8:9], 0, v1
	v_mbcnt_lo_u32_b32 v1, -1, 0
	v_readfirstlane_b32 s25, v3
	s_or_b32 s55, s20, -2.0
	s_sub_i32 s20, 0, s17
	s_load_dwordx2 s[44:45], s[0:1], 0x48
	s_load_dwordx2 s[46:47], s[0:1], 0x28
	v_mbcnt_hi_u32_b32 v1, -1, v1
	v_and_b32_e32 v3, 7, v0
	s_load_dwordx4 s[28:31], s[0:1], 0xb0
	s_load_dwordx2 s[50:51], s[0:1], 0xc8
	s_load_dword s19, s[0:1], 0x90
	s_load_dwordx2 s[52:53], s[0:1], 0xa0
	s_nop 0
	s_load_dwordx2 s[0:1], s[0:1], 0xe8
	s_mul_i32 s20, s20, s25
	v_and_or_b32 v5, v1, 64, v3
	v_and_b32_e32 v26, 48, v0
	v_lshlrev_b32_e32 v1, 1, v0
	s_mul_hi_u32 s20, s25, s20
	v_cmp_gt_u32_e64 s[10:11], 32, v0
	v_lshlrev_b32_e32 v118, 2, v0
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s29, s36, s38
	v_lshl_add_u64 v[42:43], s[50:51], 0, v[26:27]
	v_lshlrev_b32_e32 v26, 2, v2
	v_mad_u64_u32 v[44:45], s[36:37], v2, s38, v[24:25]
	v_and_b32_e32 v0, 4, v1
	s_or_b32 s57, s23, -2.0
	s_ashr_i32 s31, s3, 31
	s_add_i32 s61, s25, s20
	v_mov_b32_e32 v2, v27
	v_mov_b32_e32 v3, v27
	v_mad_u64_u32 v[46:47], s[36:37], v4, s38, v[24:25]
	v_lshl_add_u32 v48, s38, 6, v44
	v_mad_u64_u32 v[50:51], s[36:37], v6, s38, v[24:25]
	v_mov_b32_e32 v120, v0
	s_add_u32 s58, s40, 0x100
	v_mov_b32_e32 v0, v27
	v_mov_b32_e32 v1, v27
	v_lshlrev_b32_e32 v121, 2, v5
	v_lshlrev_b32_e32 v52, 2, v4
	v_lshlrev_b32_e32 v54, 2, v6
	v_mov_b64_e32 v[6:7], v[2:3]
	v_xor_b32_e32 v74, v20, v19
	v_cmp_gt_u32_e64 s[12:13], 64, v21
	v_cmp_gt_u32_e64 s[14:15], 8, v22
	v_cmp_gt_u32_e64 s[4:5], 16, v23
	v_ashrrev_i32_e32 v45, 31, v44
	v_ashrrev_i32_e32 v47, 31, v46
	v_ashrrev_i32_e32 v49, 31, v48
	v_ashrrev_i32_e32 v51, 31, v50
	s_mul_i32 s38, s18, s16
	v_add_u32_e32 v119, s24, v8
	s_mov_b32 s54, s21
	s_mov_b32 s56, s21
	s_addc_u32 s59, s41, 0
	s_mov_b32 s23, 0x110000
	s_mov_b32 s60, 0xbfb8aa3b
	s_movk_i32 s68, 0x7fff
	v_mov_b64_e32 v[4:5], v[0:1]
                                        ; implicit-def: $vgpr56
	s_branch .LBB0_3
.LBB0_2:                                ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[20:21]
	s_load_dword s20, s[48:49], 0x0
	s_waitcnt lgkmcnt(0)
	s_add_i32 s2, s20, s2
	s_cmp_ge_i32 s2, s33
	s_cbranch_scc1 .LBB0_43
.LBB0_3:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_22 Depth 2
	s_abs_i32 s21, s2
	s_mul_hi_u32 s24, s21, s61
	s_mul_i32 s25, s24, s17
	s_ashr_i32 s20, s2, 31
	s_sub_i32 s21, s21, s25
	s_xor_b32 s20, s20, s31
	s_add_i32 s25, s24, 1
	s_sub_i32 s27, s21, s17
	s_cmp_ge_u32 s21, s17
	s_cselect_b32 s24, s25, s24
	s_cselect_b32 s21, s27, s21
	s_add_i32 s25, s24, 1
	s_cmp_ge_u32 s21, s17
	s_cselect_b32 s21, s25, s24
	s_xor_b32 s21, s21, s20
	s_sub_i32 s69, s21, s20
	s_mul_i32 s20, s69, s3
	s_sub_i32 s20, s2, s20
	s_ashr_i32 s21, s20, 31
	s_lshl_b64 s[24:25], s[20:21], 2
	s_add_u32 s24, s0, s24
	s_addc_u32 s25, s1, s25
	s_lshl_b32 s62, s20, 5
	s_ashr_i32 s63, s62, 31
	global_load_dword v8, v27, s[24:25]
	s_lshl_b64 s[24:25], s[62:63], 2
	s_add_u32 s64, s50, s24
	s_addc_u32 s65, s51, s25
	global_load_dword v9, v26, s[64:65]
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s63, v8
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v8, 0xffffff, v9
	v_cmp_ne_u32_e32 vcc, s16, v8
                                        ; implicit-def: $vgpr10_vgpr11
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_5
; %bb.4:                                ;   in Loop: Header=BB0_3 Depth=1
	v_mad_u64_u32 v[8:9], s[36:37], v8, s18, v[24:25]
	v_ashrrev_i32_e32 v9, 31, v8
	v_lshl_add_u64 v[8:9], s[40:41], 0, v[8:9]
	;;#ASMSTART
	global_load_dwordx4 v[8:11], v[8:9], off

	;;#ASMEND
.LBB0_5:                                ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[20:21]
	global_load_dword v12, v52, s[64:65]
                                        ; implicit-def: $vgpr14_vgpr15
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v12, 0xffffff, v12
	v_cmp_ne_u32_e32 vcc, s16, v12
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_7
; %bb.6:                                ;   in Loop: Header=BB0_3 Depth=1
	v_mad_u64_u32 v[12:13], s[36:37], v12, s18, v[24:25]
	v_ashrrev_i32_e32 v13, 31, v12
	v_lshl_add_u64 v[12:13], s[40:41], 0, v[12:13]
	;;#ASMSTART
	global_load_dwordx4 v[12:15], v[12:13], off

	;;#ASMEND
.LBB0_7:                                ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[20:21]
	v_lshl_add_u64 v[58:59], s[64:65], 0, v[26:27]
	global_load_dword v16, v[58:59], off offset:256
                                        ; implicit-def: $vgpr18_vgpr19
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v16, 0xffffff, v16
	v_cmp_ne_u32_e32 vcc, s16, v16
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_9
; %bb.8:                                ;   in Loop: Header=BB0_3 Depth=1
	v_mad_u64_u32 v[16:17], s[36:37], v16, s18, v[24:25]
	v_ashrrev_i32_e32 v17, 31, v16
	v_lshl_add_u64 v[16:17], s[40:41], 0, v[16:17]
	;;#ASMSTART
	global_load_dwordx4 v[16:19], v[16:17], off

	;;#ASMEND
.LBB0_9:                                ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[20:21]
	global_load_dword v20, v54, s[64:65]
                                        ; implicit-def: $vgpr22_vgpr23
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v20, 0xffffff, v20
	v_cmp_ne_u32_e32 vcc, s16, v20
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_11
; %bb.10:                               ;   in Loop: Header=BB0_3 Depth=1
	v_mad_u64_u32 v[20:21], s[36:37], v20, s18, v[24:25]
	v_ashrrev_i32_e32 v21, 31, v20
	v_lshl_add_u64 v[20:21], s[40:41], 0, v[20:21]
	;;#ASMSTART
	global_load_dwordx4 v[20:23], v[20:21], off

	;;#ASMEND
.LBB0_11:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[20:21]
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	global_load_dword v53, v[58:59], off
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v53, 0xffffff, v53
	v_cmp_ne_u32_e32 vcc, s16, v53
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_13
; %bb.12:                               ;   in Loop: Header=BB0_3 Depth=1
	;;#ASMSTART
	ds_write_b64 v25, v[8:9]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v29, v[10:11]

	;;#ASMEND
.LBB0_13:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[20:21]
	v_mov_b32_e32 v53, v27
	v_lshl_add_u64 v[8:9], s[64:65], 0, v[52:53]
	global_load_dword v8, v[8:9], off
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v8, 0xffffff, v8
	v_cmp_ne_u32_e32 vcc, s16, v8
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_15
; %bb.14:                               ;   in Loop: Header=BB0_3 Depth=1
	;;#ASMSTART
	ds_write_b64 v60, v[12:13]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v61, v[14:15]

	;;#ASMEND
.LBB0_15:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[20:21]
	global_load_dword v8, v[58:59], off offset:256
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v8, 0xffffff, v8
	v_cmp_ne_u32_e32 vcc, s16, v8
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_17
; %bb.16:                               ;   in Loop: Header=BB0_3 Depth=1
	;;#ASMSTART
	ds_write_b64 v62, v[16:17]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v63, v[18:19]

	;;#ASMEND
.LBB0_17:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[20:21]
	v_mov_b32_e32 v55, v27
	v_lshl_add_u64 v[8:9], s[64:65], 0, v[54:55]
	global_load_dword v8, v[8:9], off
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v8, 0xffffff, v8
	v_cmp_ne_u32_e32 vcc, s16, v8
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_19
; %bb.18:                               ;   in Loop: Header=BB0_3 Depth=1
	;;#ASMSTART
	ds_write_b64 v64, v[20:21]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v65, v[22:23]

	;;#ASMEND
.LBB0_19:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[20:21]
	s_mul_i32 s20, s69, s22
	s_mul_i32 s27, s63, s29
	s_add_i32 s27, s27, s20
	s_ashr_i32 s21, s27, 31
	s_add_u32 s20, s44, s27
	s_addc_u32 s21, s45, s21
	v_lshl_add_u64 v[8:9], s[20:21], 0, v[44:45]
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	global_load_dwordx4 v[8:11], v[8:9], off

	;;#ASMEND
	v_lshl_add_u64 v[12:13], s[20:21], 0, v[46:47]
	;;#ASMSTART
	global_load_dwordx4 v[12:15], v[12:13], off

	;;#ASMEND
	v_lshl_add_u64 v[16:17], s[20:21], 0, v[48:49]
	;;#ASMSTART
	global_load_dwordx4 v[16:19], v[16:17], off

	;;#ASMEND
	v_lshl_add_u64 v[20:21], s[20:21], 0, v[50:51]
	;;#ASMSTART
	global_load_dwordx4 v[20:23], v[20:21], off

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v66, v[8:9]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v67, v[10:11]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v68, v[12:13]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v69, v[14:15]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v70, v[16:17]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v71, v[18:19]

	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	ds_write_b64 v72, v[20:21]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v73, v[22:23]

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_barrier
	s_and_saveexec_b64 s[20:21], s[6:7]
	s_cbranch_execz .LBB0_21
; %bb.20:                               ;   in Loop: Header=BB0_3 Depth=1
	s_barrier
.LBB0_21:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[20:21]
	v_lshl_add_u64 v[8:9], s[64:65], 0, v[26:27]
	s_addk_i32 s27, 0x100
	s_mov_b32 s70, 27
	s_mov_b64 s[66:67], s[58:59]
.LBB0_22:                               ;   Parent Loop BB0_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_ashr_i32 s21, s27, 31
	s_add_u32 s20, s44, s27
	s_addc_u32 s21, s45, s21
	s_or_b64 s[36:37], s[20:21], s[54:55]
	s_and_b64 s[72:73], s[34:35], exec
	s_cselect_b32 s21, s21, s37
	s_cselect_b32 s20, s20, s36
	buffer_load_dwordx4 v[10:13], v44, s[20:23], 0 offen
	buffer_load_dwordx4 v[14:17], v46, s[20:23], 0 offen
	buffer_load_dwordx4 v[18:21], v48, s[20:23], 0 offen
	buffer_load_dwordx4 v[122:125], v50, s[20:23], 0 offen
	;;#ASMSTART
	ds_read_b64 v[22:23], v82 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[58:59], v84 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[130:131], v86 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[132:133], v88 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[126:127], v74 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[128:129], v76 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[134:135], v78 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[136:137], v80 offset:0

	;;#ASMEND
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x32_fp8_fp8 v[4:7], v[22:23], v[126:127], v[4:7]
	v_mfma_f32_16x16x32_fp8_fp8 v[4:7], v[58:59], v[128:129], v[4:7]
	v_mfma_f32_16x16x32_fp8_fp8 v[4:7], v[22:23], v[134:135], v[4:7]
	v_mfma_f32_16x16x32_fp8_fp8 v[4:7], v[58:59], v[136:137], v[4:7]
	v_mfma_f32_16x16x32_fp8_fp8 v[4:7], v[130:131], v[126:127], v[4:7]
	v_mfma_f32_16x16x32_fp8_fp8 v[4:7], v[132:133], v[128:129], v[4:7]
	v_mfma_f32_16x16x32_fp8_fp8 v[4:7], v[130:131], v[134:135], v[4:7]
	v_mfma_f32_16x16x32_fp8_fp8 v[4:7], v[132:133], v[136:137], v[4:7]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64 v[126:127], v75 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[128:129], v77 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[134:135], v79 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[136:137], v81 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[138:139], v83 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[140:141], v85 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[142:143], v87 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[144:145], v89 offset:0

	;;#ASMEND
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x32_fp8_fp8 v[4:7], v[138:139], v[126:127], v[4:7]
	v_mfma_f32_16x16x32_fp8_fp8 v[4:7], v[140:141], v[128:129], v[4:7]
	v_mfma_f32_16x16x32_fp8_fp8 v[4:7], v[138:139], v[134:135], v[4:7]
	v_mfma_f32_16x16x32_fp8_fp8 v[4:7], v[140:141], v[136:137], v[4:7]
	v_mfma_f32_16x16x32_fp8_fp8 v[4:7], v[142:143], v[126:127], v[4:7]
	v_mfma_f32_16x16x32_fp8_fp8 v[4:7], v[144:145], v[128:129], v[4:7]
	v_mfma_f32_16x16x32_fp8_fp8 v[4:7], v[142:143], v[134:135], v[4:7]
	v_mfma_f32_16x16x32_fp8_fp8 v[4:7], v[144:145], v[136:137], v[4:7]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	global_load_dword v53, v[8:9], off
	s_or_b64 s[20:21], s[66:67], s[56:57]
	s_and_b64 s[36:37], s[42:43], exec
	s_cselect_b32 s37, s67, s21
	s_cselect_b32 s36, s66, s20
	s_mov_b32 s39, s23
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v53, 0xffffff, v53
	v_mad_u64_u32 v[126:127], s[20:21], v53, s18, v[24:25]
	buffer_load_dwordx4 v[126:129], v126, s[36:39], 0 offen
	;;#ASMSTART
	ds_read_b64 v[134:135], v90 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[136:137], v92 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[138:139], v94 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[140:141], v96 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[142:143], v91 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[144:145], v93 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[146:147], v95 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[148:149], v97 offset:0

	;;#ASMEND
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[22:23], v[134:135], v[0:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[58:59], v[136:137], v[0:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[22:23], v[138:139], v[0:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[58:59], v[140:141], v[0:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[130:131], v[134:135], v[0:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[132:133], v[136:137], v[0:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[130:131], v[138:139], v[0:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[132:133], v[140:141], v[0:3]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_write_b64 v99, v[126:127]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v98, v[128:129]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v66, v[10:11]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v67, v[12:13]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v68, v[14:15]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v69, v[16:17]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v70, v[18:19]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v71, v[20:21]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v72, v[122:123]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v73, v[124:125]

	;;#ASMEND
	s_barrier
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[22:23], v[142:143], v[0:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[58:59], v[144:145], v[0:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[22:23], v[146:147], v[0:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[58:59], v[148:149], v[0:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[130:131], v[142:143], v[0:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[132:133], v[144:145], v[0:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[130:131], v[146:147], v[0:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[132:133], v[148:149], v[0:3]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_i32 s70, s70, -1
	s_add_u32 s66, s66, 0x100
	s_addc_u32 s67, s67, 0
	s_addk_i32 s27, 0x100
	s_cmp_eq_u32 s70, 0
	s_cbranch_scc0 .LBB0_22
; %bb.23:                               ;   in Loop: Header=BB0_3 Depth=1
	; sched_barrier mask(0x00000000)
	s_and_saveexec_b64 s[20:21], s[10:11]
	s_cbranch_execz .LBB0_25
; %bb.24:                               ;   in Loop: Header=BB0_3 Depth=1
	global_load_dword v8, v118, s[64:65]
	s_add_u32 s24, s46, s24
	s_addc_u32 s25, s47, s25
	s_or_b32 s25, s25, 0xc0040000
	s_mov_b32 s27, s23
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v8, 2, v8
	v_and_b32_e32 v8, 0x3fffffc, v8
	buffer_load_dword v8, v8, s[24:27], 0 offen
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_write_b32 v119, v8

	;;#ASMEND
.LBB0_25:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[20:21]
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[12:13], v110 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[14:15], v112 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[8:9], v114 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[10:11], v116 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[16:17], v102 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[18:19], v104 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[20:21], v106 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[22:23], v108 offset:0

	;;#ASMEND
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x32_fp8_fp8 v[4:7], v[12:13], v[16:17], v[4:7]
	v_mfma_f32_16x16x32_fp8_fp8 v[4:7], v[14:15], v[18:19], v[4:7]
	v_mfma_f32_16x16x32_fp8_fp8 v[4:7], v[12:13], v[20:21], v[4:7]
	v_mfma_f32_16x16x32_fp8_fp8 v[4:7], v[14:15], v[22:23], v[4:7]
	v_mfma_f32_16x16x32_fp8_fp8 v[4:7], v[8:9], v[16:17], v[4:7]
	v_mfma_f32_16x16x32_fp8_fp8 v[4:7], v[10:11], v[18:19], v[4:7]
	v_mfma_f32_16x16x32_fp8_fp8 v[4:7], v[8:9], v[20:21], v[4:7]
	v_mfma_f32_16x16x32_fp8_fp8 v[4:7], v[10:11], v[22:23], v[4:7]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_and_saveexec_b64 s[20:21], s[12:13]
	s_cbranch_execnz .LBB0_41
; %bb.26:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[20:21]
	s_and_saveexec_b64 s[20:21], s[14:15]
	s_cbranch_execnz .LBB0_42
.LBB0_27:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[20:21]
	s_and_saveexec_b64 s[20:21], s[14:15]
	s_cbranch_execz .LBB0_29
.LBB0_28:                               ;   in Loop: Header=BB0_3 Depth=1
	v_lshl_add_u64 v[16:17], v[36:37], 0, 4
	flat_load_dword v57, v[16:17]
.LBB0_29:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[20:21]
	;;#ASMSTART
	ds_read_b64 v[16:17], v103 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[18:19], v105 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[20:21], v107 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[22:23], v109 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[58:59], v111 offset:0

	;;#ASMEND
	s_waitcnt vmcnt(0) lgkmcnt(0)
	ds_bpermute_b32 v56, v121, v56
	ds_bpermute_b32 v57, v121, v57
	;;#ASMSTART
	ds_read_b64 v[122:123], v113 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[124:125], v115 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[126:127], v117 offset:0

	;;#ASMEND
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x32_fp8_fp8 v[4:7], v[58:59], v[16:17], v[4:7]
	v_mfma_f32_16x16x32_fp8_fp8 v[4:7], v[122:123], v[18:19], v[4:7]
	v_mfma_f32_16x16x32_fp8_fp8 v[4:7], v[58:59], v[20:21], v[4:7]
	v_mfma_f32_16x16x32_fp8_fp8 v[4:7], v[122:123], v[22:23], v[4:7]
	v_mfma_f32_16x16x32_fp8_fp8 v[4:7], v[124:125], v[16:17], v[4:7]
	v_mfma_f32_16x16x32_fp8_fp8 v[4:7], v[126:127], v[18:19], v[4:7]
	v_mfma_f32_16x16x32_fp8_fp8 v[4:7], v[124:125], v[20:21], v[4:7]
	v_mfma_f32_16x16x32_fp8_fp8 v[4:7], v[126:127], v[22:23], v[4:7]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_and_saveexec_b64 s[20:21], s[4:5]
	s_cbranch_execz .LBB0_31
; %bb.30:                               ;   in Loop: Header=BB0_3 Depth=1
	flat_load_dword v16, v[38:39]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	scratch_store_dword v120, v16, off
	flat_load_dword v16, v[40:41]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	scratch_store_dword v120, v16, off offset:4
.LBB0_31:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[20:21]
	;;#ASMSTART
	ds_read_b64 v[16:17], v90 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[18:19], v92 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[20:21], v94 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[22:23], v96 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[58:59], v91 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[122:123], v93 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[124:125], v95 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[126:127], v97 offset:0

	;;#ASMEND
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[12:13], v[16:17], v[0:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[14:15], v[18:19], v[0:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[12:13], v[20:21], v[0:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[14:15], v[22:23], v[0:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[8:9], v[16:17], v[0:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[10:11], v[18:19], v[0:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[8:9], v[20:21], v[0:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[10:11], v[22:23], v[0:3]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[12:13], v[58:59], v[0:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[14:15], v[122:123], v[0:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[12:13], v[124:125], v[0:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[14:15], v[126:127], v[0:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[8:9], v[58:59], v[0:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[10:11], v[122:123], v[0:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[8:9], v[124:125], v[0:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[10:11], v[126:127], v[0:3]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	scratch_load_dwordx2 v[8:9], off, off
	s_barrier
	; sched_barrier mask(0x00000000)
	s_and_saveexec_b64 s[20:21], s[8:9]
	s_cbranch_execz .LBB0_33
; %bb.32:                               ;   in Loop: Header=BB0_3 Depth=1
	s_barrier
.LBB0_33:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[20:21]
	v_add_u32_e32 v10, s62, v100
	v_ashrrev_i32_e32 v11, 31, v10
	v_lshl_add_u64 v[10:11], v[10:11], 2, v[42:43]
	global_load_dword v14, v[10:11], off
	s_waitcnt lgkmcnt(0)
	v_pk_mul_f32 v[4:5], v[4:5], v[56:57]
	v_pk_mul_f32 v[0:1], v[0:1], v[56:57]
	s_waitcnt vmcnt(1)
	v_pk_mul_f32 v[4:5], v[4:5], v[8:9] op_sel_hi:[1,0]
	v_pk_mul_f32 v[0:1], v[0:1], v[8:9] op_sel:[0,1]
	v_pk_mul_f32 v[12:13], v[4:5], s[60:61] op_sel_hi:[1,0]
	s_nop 0
	v_exp_f32_e32 v12, v12
	v_exp_f32_e32 v13, v13
	s_nop 0
	v_pk_add_f32 v[12:13], v[12:13], 1.0 op_sel_hi:[1,0]
	s_nop 0
	v_div_scale_f32 v15, s[20:21], v13, v13, 1.0
	v_rcp_f32_e32 v16, v15
	s_nop 0
	v_fma_f32 v17, -v15, v16, 1.0
	v_fmac_f32_e32 v16, v17, v16
	v_div_scale_f32 v17, vcc, 1.0, v13, 1.0
	v_mul_f32_e32 v18, v17, v16
	v_fma_f32 v19, -v15, v18, v17
	v_fmac_f32_e32 v18, v19, v16
	v_fma_f32 v15, -v15, v18, v17
	v_div_scale_f32 v17, s[20:21], v12, v12, 1.0
	v_rcp_f32_e32 v19, v17
	v_div_fmas_f32 v15, v15, v16, v18
	v_div_fixup_f32 v13, v15, v13, 1.0
	v_fma_f32 v15, -v17, v19, 1.0
	v_fmac_f32_e32 v19, v15, v19
	v_div_scale_f32 v15, vcc, 1.0, v12, 1.0
	v_mul_f32_e32 v16, v15, v19
	v_fma_f32 v18, -v17, v16, v15
	v_fmac_f32_e32 v16, v18, v19
	v_fma_f32 v15, -v17, v16, v15
	v_div_fmas_f32 v15, v15, v19, v16
	v_div_fixup_f32 v12, v15, v12, 1.0
	v_pk_mul_f32 v[4:5], v[4:5], v[12:13]
	v_lshl_or_b32 v12, s69, 6, v101
	v_ashrrev_i32_e32 v13, 31, v12
	v_pk_mul_f32 v[4:5], v[4:5], v[0:1]
	v_lshl_add_u64 v[12:13], v[12:13], 1, s[52:53]
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v15, 0xffffff, v14
	v_cmp_ne_u32_e32 vcc, s28, v15
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_35
; %bb.34:                               ;   in Loop: Header=BB0_3 Depth=1
	v_lshrrev_b32_e32 v14, 24, v14
	v_lshl_add_u32 v14, v15, 1, v14
	v_mad_u64_u32 v[14:15], s[24:25], v14, s30, v[28:29]
	v_bfe_u32 v16, v4, 16, 1
	v_ashrrev_i32_e32 v15, 31, v14
	v_add3_u32 v16, v16, v4, s68
	v_or_b32_e32 v17, 0x400000, v4
	v_cmp_u_f32_e32 vcc, v4, v4
	v_lshl_add_u64 v[14:15], v[14:15], 1, v[12:13]
	s_nop 0
	v_cndmask_b32_e32 v16, v16, v17, vcc
	global_store_short_d16_hi v[14:15], v16, off
.LBB0_35:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[20:21]
	global_load_dword v14, v[10:11], off offset:4
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v15, 0xffffff, v14
	v_cmp_ne_u32_e32 vcc, s28, v15
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_37
; %bb.36:                               ;   in Loop: Header=BB0_3 Depth=1
	v_lshrrev_b32_e32 v14, 24, v14
	v_lshl_add_u32 v14, v15, 1, v14
	v_mad_u64_u32 v[14:15], s[24:25], v14, s30, v[28:29]
	v_bfe_u32 v16, v5, 16, 1
	v_ashrrev_i32_e32 v15, 31, v14
	v_add3_u32 v16, v16, v5, s68
	v_or_b32_e32 v17, 0x400000, v5
	v_cmp_u_f32_e32 vcc, v5, v5
	v_lshl_add_u64 v[14:15], v[14:15], 1, v[12:13]
	s_nop 0
	v_cndmask_b32_e32 v16, v16, v17, vcc
	global_store_short_d16_hi v[14:15], v16, off
.LBB0_37:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[20:21]
	global_load_dword v14, v[10:11], off offset:8
	v_mov_b32_e32 v16, v8
	v_mov_b32_e32 v17, v8
	v_pk_mul_f32 v[6:7], v[6:7], 0 op_sel_hi:[1,0]
	v_mov_b32_e32 v8, v9
	v_pk_mul_f32 v[6:7], v[6:7], v[16:17]
	v_pk_mul_f32 v[2:3], v[2:3], 0 op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[6:7], s[60:61] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[8:9]
	v_exp_f32_e32 v16, v16
	v_exp_f32_e32 v17, v17
	s_nop 0
	v_pk_add_f32 v[16:17], v[16:17], 1.0 op_sel_hi:[1,0]
	s_nop 0
	v_div_scale_f32 v15, s[20:21], v17, v17, 1.0
	v_rcp_f32_e32 v18, v15
	s_nop 0
	v_fma_f32 v8, -v15, v18, 1.0
	v_fmac_f32_e32 v18, v8, v18
	v_div_scale_f32 v8, vcc, 1.0, v17, 1.0
	v_mul_f32_e32 v9, v8, v18
	v_fma_f32 v19, -v15, v9, v8
	v_fmac_f32_e32 v9, v19, v18
	v_fma_f32 v8, -v15, v9, v8
	v_div_scale_f32 v15, s[20:21], v16, v16, 1.0
	v_rcp_f32_e32 v19, v15
	v_div_fmas_f32 v8, v8, v18, v9
	v_div_fixup_f32 v9, v8, v17, 1.0
	v_fma_f32 v8, -v15, v19, 1.0
	v_fmac_f32_e32 v19, v8, v19
	v_div_scale_f32 v8, vcc, 1.0, v16, 1.0
	v_mul_f32_e32 v17, v8, v19
	v_fma_f32 v18, -v15, v17, v8
	v_fmac_f32_e32 v17, v18, v19
	v_fma_f32 v8, -v15, v17, v8
	v_div_fmas_f32 v8, v8, v19, v17
	v_div_fixup_f32 v8, v8, v16, 1.0
	v_pk_mul_f32 v[6:7], v[6:7], v[8:9]
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v8, 0xffffff, v14
	v_pk_mul_f32 v[6:7], v[6:7], v[2:3]
	v_cmp_ne_u32_e32 vcc, s28, v8
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_39
; %bb.38:                               ;   in Loop: Header=BB0_3 Depth=1
	v_lshrrev_b32_e32 v9, 24, v14
	v_lshl_add_u32 v8, v8, 1, v9
	v_mad_u64_u32 v[8:9], s[24:25], v8, s30, v[28:29]
	v_bfe_u32 v14, v6, 16, 1
	v_ashrrev_i32_e32 v9, 31, v8
	v_add3_u32 v14, v14, v6, s68
	v_or_b32_e32 v15, 0x400000, v6
	v_cmp_u_f32_e32 vcc, v6, v6
	v_lshl_add_u64 v[8:9], v[8:9], 1, v[12:13]
	s_nop 0
	v_cndmask_b32_e32 v14, v14, v15, vcc
	global_store_short_d16_hi v[8:9], v14, off
.LBB0_39:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[20:21]
	global_load_dword v8, v[10:11], off offset:12
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v9, 0xffffff, v8
	v_cmp_ne_u32_e32 vcc, s28, v9
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_2
; %bb.40:                               ;   in Loop: Header=BB0_3 Depth=1
	v_lshrrev_b32_e32 v8, 24, v8
	v_lshl_add_u32 v8, v9, 1, v8
	v_mad_u64_u32 v[8:9], s[24:25], v8, s30, v[28:29]
	v_bfe_u32 v10, v7, 16, 1
	v_ashrrev_i32_e32 v9, 31, v8
	v_add3_u32 v10, v10, v7, s68
	v_or_b32_e32 v11, 0x400000, v7
	v_cmp_u_f32_e32 vcc, v7, v7
	v_lshl_add_u64 v[8:9], v[8:9], 1, v[12:13]
	s_nop 0
	v_cndmask_b32_e32 v10, v10, v11, vcc
	global_store_short_d16_hi v[8:9], v10, off
	s_branch .LBB0_2
.LBB0_41:                               ;   in Loop: Header=BB0_3 Depth=1
	s_mul_i32 s24, s63, s19
	s_lshl_b32 s25, s69, 6
	s_add_i32 s24, s24, s25
	s_ashr_i32 s25, s24, 31
	v_lshl_add_u64 v[16:17], s[24:25], 2, v[32:33]
	global_load_dwordx4 v[18:21], v[16:17], off
	s_waitcnt vmcnt(0)
	flat_store_dwordx4 v[30:31], v[18:21]
	global_load_dwordx4 v[16:19], v[16:17], off offset:2048
	s_waitcnt vmcnt(0)
	flat_store_dwordx4 v[34:35], v[16:19]
	s_or_b64 exec, exec, s[20:21]
	s_and_saveexec_b64 s[20:21], s[14:15]
	s_cbranch_execz .LBB0_27
.LBB0_42:                               ;   in Loop: Header=BB0_3 Depth=1
	flat_load_dword v56, v[36:37]
	s_or_b64 exec, exec, s[20:21]
	s_and_saveexec_b64 s[20:21], s[14:15]
	s_cbranch_execnz .LBB0_28
	s_branch .LBB0_29
.LBB0_43:
	s_endpgm
.Lfunc_end0:
	.size	_Z6kernel18moe_stage1_globals, .Lfunc_end0-_Z6kernel18moe_stage1_globals
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z6kernel18moe_stage1_globals
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 12
		.amdhsa_kernarg_size 536
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 150
		.amdhsa_next_free_sgpr 74
		.amdhsa_accum_offset 152
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
                                        ; -- End function
	.set .L_Z6kernel18moe_stage1_globals.num_vgpr, 150
	.set .L_Z6kernel18moe_stage1_globals.num_agpr, 0
	.set .L_Z6kernel18moe_stage1_globals.numbered_sgpr, 74
	.set .L_Z6kernel18moe_stage1_globals.num_named_barrier, 0
	.set .L_Z6kernel18moe_stage1_globals.private_seg_size, 12
	.set .L_Z6kernel18moe_stage1_globals.uses_vcc, 1
	.set .L_Z6kernel18moe_stage1_globals.uses_flat_scratch, 0
	.set .L_Z6kernel18moe_stage1_globals.has_dyn_sized_stack, 0
	.set .L_Z6kernel18moe_stage1_globals.has_recursion, 0
	.set .L_Z6kernel18moe_stage1_globals.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 5684
; TotalNumSgprs: 80
; NumVgprs: 150
; NumAgprs: 0
; TotalNumVgprs: 150
; ScratchSize: 12
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 9
; VGPRBlocks: 18
; NumSGPRsForWavesPerEU: 80
; NumVGPRsForWavesPerEU: 150
; AccumOffset: 152
; Occupancy: 3
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 1
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 37
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.text
	.type	__hip_cuid_2e999c1ab39fddd,@object ; @__hip_cuid_2e999c1ab39fddd
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_2e999c1ab39fddd
__hip_cuid_2e999c1ab39fddd:
	.byte	0                               ; 0x0
	.size	__hip_cuid_2e999c1ab39fddd, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __shm
	.addrsig_sym __hip_cuid_2e999c1ab39fddd
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .offset:         0
        .size:           280
        .value_kind:     by_value
      - .offset:         280
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         284
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         288
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         292
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         294
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         296
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         298
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         300
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         302
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         320
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         328
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         336
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         344
        .size:           2
        .value_kind:     hidden_grid_dims
      - .offset:         400
        .size:           4
        .value_kind:     hidden_dynamic_lds_size
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 536
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 512
    .name:           _Z6kernel18moe_stage1_globals
    .private_segment_fixed_size: 12
    .sgpr_count:     80
    .sgpr_spill_count: 0
    .symbol:         _Z6kernel18moe_stage1_globals.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     150
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata

# __CLANG_OFFLOAD_BUNDLE____END__ hip-amdgcn-amd-amdhsa--gfx942

# __CLANG_OFFLOAD_BUNDLE____START__ host-x86_64-unknown-linux-gnu-
	.att_syntax
	.file	"kernel.cpp"
                                        # Start of file scope inline assembly
	.globl	_ZSt21ios_base_library_initv

                                        # End of file scope inline assembly
	.text
	.globl	_Z21__device_stub__kernel18moe_stage1_globals # -- Begin function _Z21__device_stub__kernel18moe_stage1_globals
	.prefalign	4, .Lfunc_end0, nop
	.type	_Z21__device_stub__kernel18moe_stage1_globals,@function
_Z21__device_stub__kernel18moe_stage1_globals: # @_Z21__device_stub__kernel18moe_stage1_globals
	.cfi_startproc
# %bb.0:
	subq	$56, %rsp
	.cfi_def_cfa_offset 64
	movq	%rdi, (%rsp)
	leaq	40(%rsp), %rdi
	leaq	24(%rsp), %rsi
	leaq	16(%rsp), %rdx
	leaq	8(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	40(%rsp), %rsi
	movl	48(%rsp), %edx
	movq	24(%rsp), %rcx
	movl	32(%rsp), %r8d
	movq	_Z6kernel18moe_stage1_globals@GOTPCREL(%rip), %rdi
	movq	%rsp, %r9
	pushq	8(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$72, %rsp
	.cfi_adjust_cfa_offset -72
	retq
.Lfunc_end0:
	.size	_Z21__device_stub__kernel18moe_stage1_globals, .Lfunc_end0-_Z21__device_stub__kernel18moe_stage1_globals
	.cfi_endproc
                                        # -- End function
	.globl	_Z4call18moe_stage1_globals     # -- Begin function _Z4call18moe_stage1_globals
	.prefalign	4, .Lfunc_end1, nop
	.type	_Z4call18moe_stage1_globals,@function
_Z4call18moe_stage1_globals:            # @_Z4call18moe_stage1_globals
	.cfi_startproc
# %bb.0:
	pushq	%rbx
	.cfi_def_cfa_offset 16
	subq	$1808, %rsp                     # imm = 0x710
	.cfi_def_cfa_offset 1824
	.cfi_offset %rbx, -16
	movq	%rdi, %rbx
	movq	_Z6kernel18moe_stage1_globals@GOTPCREL(%rip), %rdi
	movl	$8, %esi
	movl	$40960, %edx                    # imm = 0xA000
	callq	hipFuncSetAttribute@PLT
	leaq	336(%rsp), %rdi
	xorl	%esi, %esi
	callq	hipGetDevicePropertiesR0600@PLT
	movl	724(%rsp), %edi
	movabsq	$4294967296, %rdx               # imm = 0x100000000
	orq	%rdx, %rdi
	movq	272(%rbx), %r9
	orq	$512, %rdx                      # imm = 0x200
	movl	$40960, %r8d                    # imm = 0xA000
	movl	$1, %esi
	movl	$1, %ecx
	callq	__hipPushCallConfiguration@PLT
	testl	%eax, %eax
	je	.LBB1_1
# %bb.2:
	addq	$1808, %rsp                     # imm = 0x710
	.cfi_def_cfa_offset 16
	popq	%rbx
	.cfi_def_cfa_offset 8
	retq
.LBB1_1:
	.cfi_def_cfa_offset 1824
	movq	(%rbx), %rax
	movq	%rax, 56(%rsp)
	movups	16(%rbx), %xmm0
	movups	%xmm0, 72(%rsp)
	movq	40(%rbx), %rax
	movq	%rax, 96(%rsp)
	movq	56(%rbx), %rax
	movq	%rax, 112(%rsp)
	movq	72(%rbx), %rax
	movq	%rax, 128(%rsp)
	movups	88(%rbx), %xmm0
	movups	%xmm0, 144(%rsp)
	movq	104(%rbx), %rax
	movq	%rax, 160(%rsp)
	movq	120(%rbx), %rax
	movq	%rax, 176(%rsp)
	movups	136(%rbx), %xmm0
	movups	%xmm0, 192(%rsp)
	movq	160(%rbx), %rax
	movq	%rax, 216(%rsp)
	movups	176(%rbx), %xmm0
	movups	%xmm0, 232(%rsp)
	movq	200(%rbx), %rax
	movq	%rax, 256(%rsp)
	movq	216(%rbx), %rax
	movq	%rax, 272(%rsp)
	movq	232(%rbx), %rax
	movq	%rax, 288(%rsp)
	movq	248(%rbx), %rax
	movq	%rax, 304(%rsp)
	movups	264(%rbx), %xmm0
	movups	%xmm0, 320(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, (%rsp)
	leaq	40(%rsp), %rdi
	leaq	24(%rsp), %rsi
	leaq	16(%rsp), %rdx
	leaq	8(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	40(%rsp), %rsi
	movl	48(%rsp), %edx
	movq	24(%rsp), %rcx
	movl	32(%rsp), %r8d
	movq	_Z6kernel18moe_stage1_globals@GOTPCREL(%rip), %rdi
	movq	%rsp, %r9
	pushq	8(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
	addq	$1808, %rsp                     # imm = 0x710
	.cfi_def_cfa_offset 16
	popq	%rbx
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end1:
	.size	_Z4call18moe_stage1_globals, .Lfunc_end1-_Z4call18moe_stage1_globals
	.cfi_endproc
                                        # -- End function
	.prefalign	4, .Lfunc_end2, nop     # -- Begin function __hip_module_ctor
	.type	__hip_module_ctor,@function
__hip_module_ctor:                      # @__hip_module_ctor
	.cfi_startproc
# %bb.0:
	subq	$40, %rsp
	.cfi_def_cfa_offset 48
	movq	__hip_gpubin_handle_2e999c1ab39fddd(%rip), %rdi
	testq	%rdi, %rdi
	jne	.LBB2_2
# %bb.1:
	leaq	__hip_fatbin_wrapper(%rip), %rdi
	callq	__hipRegisterFatBinary@PLT
	movq	%rax, %rdi
	movq	%rax, __hip_gpubin_handle_2e999c1ab39fddd(%rip)
.LBB2_2:
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movq	_Z6kernel18moe_stage1_globals@GOTPCREL(%rip), %rsi
	leaq	.L__unnamed_1(%rip), %rcx
	movq	%rcx, %rdx
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction@PLT
	leaq	__hip_module_dtor(%rip), %rdi
	addq	$40, %rsp
	.cfi_def_cfa_offset 8
	jmp	atexit@PLT                      # TAILCALL
.Lfunc_end2:
	.size	__hip_module_ctor, .Lfunc_end2-__hip_module_ctor
	.cfi_endproc
                                        # -- End function
	.prefalign	4, .Lfunc_end3, nop     # -- Begin function __hip_module_dtor
	.type	__hip_module_dtor,@function
__hip_module_dtor:                      # @__hip_module_dtor
	.cfi_startproc
# %bb.0:
	movq	__hip_gpubin_handle_2e999c1ab39fddd(%rip), %rdi
	testq	%rdi, %rdi
	je	.LBB3_2
# %bb.1:
	pushq	%rax
	.cfi_def_cfa_offset 16
	callq	__hipUnregisterFatBinary@PLT
	movq	$0, __hip_gpubin_handle_2e999c1ab39fddd(%rip)
	addq	$8, %rsp
	.cfi_def_cfa_offset 8
.LBB3_2:
	retq
.Lfunc_end3:
	.size	__hip_module_dtor, .Lfunc_end3-__hip_module_dtor
	.cfi_endproc
                                        # -- End function
	.type	_Z6kernel18moe_stage1_globals,@object # @_Z6kernel18moe_stage1_globals
	.section	.data.rel.ro,"aw",@progbits
	.globl	_Z6kernel18moe_stage1_globals
	.p2align	3, 0x0
_Z6kernel18moe_stage1_globals:
	.quad	_Z21__device_stub__kernel18moe_stage1_globals
	.size	_Z6kernel18moe_stage1_globals, 8

	.type	.L__unnamed_1,@object           # @0
	.section	.rodata.str1.1,"aMS",@progbits,1
.L__unnamed_1:
	.asciz	"_Z6kernel18moe_stage1_globals"
	.size	.L__unnamed_1, 30

	.type	__hip_fatbin_wrapper,@object    # @__hip_fatbin_wrapper
	.section	.hipFatBinSegment,"aw",@progbits
	.p2align	3, 0x0
__hip_fatbin_wrapper:
	.long	1212764230                      # 0x48495046
	.long	1                               # 0x1
	.quad	__hip_fatbin_2e999c1ab39fddd
	.quad	0
	.size	__hip_fatbin_wrapper, 24

	.type	__hip_gpubin_handle_2e999c1ab39fddd,@object # @__hip_gpubin_handle_2e999c1ab39fddd
	.local	__hip_gpubin_handle_2e999c1ab39fddd
	.comm	__hip_gpubin_handle_2e999c1ab39fddd,8,8
	.section	.init_array,"aw",@init_array
	.p2align	3, 0x0
	.quad	__hip_module_ctor
	.type	__hip_cuid_2e999c1ab39fddd,@object # @__hip_cuid_2e999c1ab39fddd
	.bss
	.globl	__hip_cuid_2e999c1ab39fddd
__hip_cuid_2e999c1ab39fddd:
	.byte	0                               # 0x0
	.size	__hip_cuid_2e999c1ab39fddd, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym _Z21__device_stub__kernel18moe_stage1_globals
	.addrsig_sym __hip_module_ctor
	.addrsig_sym __hip_module_dtor
	.addrsig_sym _Z6kernel18moe_stage1_globals
	.addrsig_sym __hip_fatbin_2e999c1ab39fddd
	.addrsig_sym __hip_fatbin_wrapper
	.addrsig_sym __hip_cuid_2e999c1ab39fddd

# __CLANG_OFFLOAD_BUNDLE____END__ host-x86_64-unknown-linux-gnu-
