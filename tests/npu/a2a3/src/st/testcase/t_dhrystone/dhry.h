/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#include <stdio.h>
#include <pto/common/constants.hpp>

#ifdef NOSTRUCTASSIGN
#define structassign(d, s) memcpy(&(d), &(s), sizeof(d))
#else
#define structassign(d, s) d = s
#endif

#ifdef NOENUM
#define Ident_1 0
#define Ident_2 1
#define Ident_3 2
#define Ident_4 3
#define Ident_5 4
typedef int Enumeration;
#else
typedef enum
{
    Ident_1,
    Ident_2,
    Ident_3,
    Ident_4,
    Ident_5
} Enumeration;
#endif
/* for boolean and enumeration types in Ada, Pascal */

/* General definitions: */
#define Null 0
/* Value of a Null pointer */
#define true 1
#define false 0

typedef int One_Thirty;
typedef int One_Fifty;
typedef char Capital_Letter;
typedef int Boolean;
typedef char Str_30[31];
typedef int Arr_1_Dim[50];
typedef int Arr_2_Dim[50][50];

typedef struct record {
    struct record *Ptr_Comp;
    Enumeration Discr;
    union {
        struct {
            Enumeration Enum_Comp;
            int Int_Comp;
            char Str_Comp[31];
        } var_1;
        struct {
            Enumeration E_Comp_2;
            char Str_2_Comp[31];
        } var_2;
        struct {
            char Ch_1_Comp;
            char Ch_2_Comp;
        } var_3;
    } variant;
} Rec_Type, *Rec_Pointer;

AICORE void Proc_1(Rec_Pointer Ptr_Val_Par);
AICORE void Proc_2(One_Fifty *Int_Par_Ref);
AICORE void Proc_3(Rec_Pointer *Ptr_Ref_Par);
AICORE void Proc_4(void);
AICORE void Proc_5(void);
AICORE void Proc_6(Enumeration Enum_Val_Par, Enumeration *Enum_Ref_Par);
AICORE void Proc_7(One_Fifty Int_1_Par_Val, One_Fifty Int_2_Par_Val, One_Fifty *Int_Par_Ref);
AICORE void Proc_8(int *Arr_1_Par_Ref, int *Arr_2_Par_Ref, int Int_1_Par_Val, int Int_2_Par_Val);
AICORE Enumeration Func_1(Capital_Letter Ch_1_Par_Val, Capital_Letter Ch_2_Par_Val);
AICORE Boolean Func_2(Str_30 Str_1_Par_Ref, Str_30 Str_2_Par_Ref);
AICORE Boolean Func_3(Enumeration Enum_Par_Val);
AICORE void memcpy(char *d, char *s, int l);
AICORE void strcpy(char *des, char *source);
AICORE int strcmp(char *sl, char *s2);