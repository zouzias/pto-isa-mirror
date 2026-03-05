/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "acl/acl.h"
#include "dhry.h"

using namespace pto;

AICORE void memcpy(char *d, char *s, int l)
{
    while (l--)
        *d++ = *s++;
}

AICORE void strcpy(char *des, char *source)
{
    char *r = des;

    if ((des == NULL) || (source == NULL))
        return;

    while ((*r++ = *source++) != '\0')
        ;
}

AICORE int strcmp(char *sl, char *s2)
{
    for (; *sl == *s2; ++sl, ++s2)
        if (*sl == '\0')
            return (0);
    return ((*(unsigned char *)sl < *(unsigned char *)s2) ? -1 : +1);
}

AICORE void Proc_1(Rec_Pointer Ptr_Val_Par)
{
    Rec_Pointer Ptr_Glob = (Rec_Pointer)0x100000;
    Rec_Pointer Next_Record = Ptr_Val_Par->Ptr_Comp;
    /* == Ptr_Glob_Next */
    /* Local variable, initialized with Ptr_Val_Par->Ptr_Comp,    */
    /* corresponds to "rename" in Ada, "with" in Pascal           */

    structassign(*Ptr_Val_Par->Ptr_Comp, *Ptr_Glob);
    Ptr_Val_Par->variant.var_1.Int_Comp = 5;
    Next_Record->variant.var_1.Int_Comp = Ptr_Val_Par->variant.var_1.Int_Comp;
    Next_Record->Ptr_Comp = Ptr_Val_Par->Ptr_Comp;
    Proc_3(&Next_Record->Ptr_Comp);
    /* Ptr_Val_Par->Ptr_Comp->Ptr_Comp
                        == Ptr_Glob->Ptr_Comp */
    if (Next_Record->Discr == Ident_1)
    /* then, executed */
    {
        Next_Record->variant.var_1.Int_Comp = 6;
        Proc_6(Ptr_Val_Par->variant.var_1.Enum_Comp, &Next_Record->variant.var_1.Enum_Comp);
        Next_Record->Ptr_Comp = Ptr_Glob->Ptr_Comp;
        Proc_7(Next_Record->variant.var_1.Int_Comp, 10, &Next_Record->variant.var_1.Int_Comp);
    } else /* not executed */
        structassign(*Ptr_Val_Par, *Ptr_Val_Par->Ptr_Comp);
} /* Proc_1 */

AICORE void Proc_2(One_Fifty *Int_Par_Ref)

{
    char *Ch_1_Glob = (char *)0x100110;
    char *Int_Glob = (char *)0x100100;
    One_Fifty Int_Loc;
    Enumeration Enum_Loc;

    Int_Loc = *Int_Par_Ref + 10;
    do /* executed once */
        if (*Ch_1_Glob == 'A')
        /* then, executed */
        {
            Int_Loc -= 1;
            *Int_Par_Ref = Int_Loc - *Int_Glob;
            Enum_Loc = Ident_1;
        }
    while (Enum_Loc != Ident_1); /* true */
} /* Proc_2 */

AICORE void Proc_3(Rec_Pointer *Ptr_Ref_Par)

{
    Rec_Pointer Ptr_Glob = (Rec_Pointer)0x100000;
    int *Int_Glob = (int *)0x100100;
    if (Ptr_Glob != Null)
        /* then, executed */
        *Ptr_Ref_Par = Ptr_Glob->Ptr_Comp;
    Proc_7(10, *Int_Glob, &Ptr_Glob->variant.var_1.Int_Comp);
} /* Proc_3 */

AICORE void Proc_4(void)
{
    Boolean Bool_Loc;

    Boolean *Bool_Glob = (Boolean *)0x100108;
    char *Ch_1_Glob = (char *)0x100110;
    char *Ch_2_Glob = (char *)0x100118;

    Bool_Loc = *Ch_1_Glob == 'A';
    *Bool_Glob = Bool_Loc | *Bool_Glob;
    *Ch_2_Glob = 'B';
} /* Proc_4 */

AICORE void Proc_5(void)
{
    Boolean *Bool_Glob = (Boolean *)0x100108;
    char *Ch_1_Glob = (char *)0x100110;

    *Ch_1_Glob = 'A';
    *Bool_Glob = false;
} /* Proc_5 */

AICORE void Proc_6(Enumeration Enum_Val_Par, Enumeration *Enum_Ref_Par)
{
    char *Int_Glob = (char *)0x100100;
    *Enum_Ref_Par = Enum_Val_Par;
    if (!Func_3(Enum_Val_Par))
        /* then, not executed */
        *Enum_Ref_Par = Ident_4;
    switch (Enum_Val_Par) {
        case Ident_1:
            *Enum_Ref_Par = Ident_1;
            break;
        case Ident_2:
            if (*Int_Glob > 100)
                /* then */
                *Enum_Ref_Par = Ident_1;
            else
                *Enum_Ref_Par = Ident_4;
            break;
        case Ident_3: /* executed */
            *Enum_Ref_Par = Ident_2;
            break;
        case Ident_4:
            break;
        case Ident_5:
            *Enum_Ref_Par = Ident_3;
            break;
    } /* switch */
} /* Proc_6 */

AICORE void Proc_7(One_Fifty Int_1_Par_Val, One_Fifty Int_2_Par_Val, One_Fifty *Int_Par_Ref)
{
    One_Fifty Int_Loc;

    Int_Loc = Int_1_Par_Val + 2;
    *Int_Par_Ref = Int_2_Par_Val + Int_Loc;
} /* Proc_7 */

AICORE void Proc_8(int *Arr_1_Par_Ref, int *Arr_2_Par_Ref, int Int_1_Par_Val, int Int_2_Par_Val)
{
    One_Fifty Int_Index;
    One_Fifty Int_Loc;
    char *Int_Glob = (char *)0x100100;

    Int_Loc = Int_1_Par_Val + 5;
    *(Arr_1_Par_Ref + Int_Loc) = Int_2_Par_Val;
    *(Arr_1_Par_Ref + Int_Loc + 1) = *(Arr_1_Par_Ref + Int_Loc);
    *(Arr_1_Par_Ref + Int_Loc + 30) = Int_Loc;
    for (Int_Index = Int_Loc; Int_Index <= Int_Loc + 1; ++Int_Index)
        *(Arr_2_Par_Ref + Int_Loc * 50 + Int_Index) = Int_Loc;
    (*(Arr_2_Par_Ref + Int_Loc * 50 + Int_Loc - 1)) += 1;
    *(Arr_2_Par_Ref + (Int_Loc + 20) * 50 + Int_Loc) = *(Arr_1_Par_Ref + Int_Loc);
    *Int_Glob = 5;
} /* Proc_8 */

AICORE Enumeration Func_1(Capital_Letter Ch_1_Par_Val, Capital_Letter Ch_2_Par_Val)
/*************************************************/
/* executed three times                                         */
/* first call:      Ch_1_Par_Val == 'H', Ch_2_Par_Val == 'R'    */
/* second call:     Ch_1_Par_Val == 'A', Ch_2_Par_Val == 'C'    */
/* third call:      Ch_1_Par_Val == 'B', Ch_2_Par_Val == 'C'    */
{
    Capital_Letter Ch_1_Loc;
    Capital_Letter Ch_2_Loc;
    char *Ch_1_Glob = (char *)0x100110;

    Ch_1_Loc = Ch_1_Par_Val;
    Ch_2_Loc = Ch_1_Loc;
    if (Ch_2_Loc != Ch_2_Par_Val)
        /* then, executed */
        return (Ident_1);
    else /* not executed */
    {
        *Ch_1_Glob = Ch_1_Loc;
        return (Ident_2);
    }
} /* Func_1 */

AICORE Boolean Func_2(Str_30 Str_1_Par_Ref, Str_30 Str_2_Par_Ref)
/*************************************************/
/* executed once */
/* Str_1_Par_Ref == "DHRYSTONE PROGRAM, 1'ST STRING" */
/* Str_2_Par_Ref == "DHRYSTONE PROGRAM, 2'ND STRING" */
{
    One_Thirty Int_Loc;
    Capital_Letter Ch_Loc;
    int *Int_Glob = (int *)0x100100;

    Int_Loc = 2;
    while (Int_Loc <= 2) /* loop body executed once */
        if (Func_1(Str_1_Par_Ref[Int_Loc], Str_2_Par_Ref[Int_Loc + 1]) == Ident_1)
        /* then, executed */
        {
            Ch_Loc = 'A';
            Int_Loc += 1;
        } /* if, while */
    if (Ch_Loc >= 'W' && Ch_Loc < 'Z')
        /* then, not executed */
        Int_Loc = 7;
    if (Ch_Loc == 'R')
        /* then, not executed */
        return (true);
    else /* executed */
    {
        if (strcmp(Str_1_Par_Ref, Str_2_Par_Ref) > 0)
        /* then, not executed */
        {
            Int_Loc += 7;
            *Int_Glob = Int_Loc;
            return (true);
        } else /* executed */
            return (false);
    } /* if Ch_Loc */
} /* Func_2 */

AICORE Boolean Func_3(Enumeration Enum_Par_Val)
/***************************/
/* executed once        */
/* Enum_Par_Val == Ident_3 */
{
    Enumeration Enum_Loc;

    Enum_Loc = Enum_Par_Val;
    if (Enum_Loc == Ident_3)
        /* then, executed */
        return (true);
    else /* not executed */
        return (false);
} /* Func_3 */

template <int iteration>
__global__ AICORE void runTDhrystone()
{
    One_Fifty Int_1_Loc;
    One_Fifty Int_2_Loc;
    One_Fifty Int_3_Loc;
    char Ch_Index;
    Enumeration Enum_Loc;
    Str_30 Str_1_Loc;
    Str_30 Str_2_Loc;
    int Run_Index;
    int Number_Of_Runs;

    Rec_Pointer Ptr_Glob = (Rec_Pointer)0x100000;
    Rec_Pointer Next_Ptr_Glob = (Rec_Pointer)0x100080;
    int *Int_Glob = (int *)0x100100;
    Boolean *Bool_Glob = (Boolean *)0x100108;
    char *Ch_2_Glob = (char *)0x100118;
    int *Arr_1_Glob = (int *)0x101000;
    int *Arr_2_Glob = (int *)0x102000;

    /* Initializations */
    char cpystr0[31] = {'D', 'H', 'R', 'Y', 'S', 'T', 'O', 'N', 'E', ' ', 'P', 'R', 'O', 'G', 'R',
                        'A', 'M', ',', ' ', 'S', 'O', 'M', 'E', ' ', 'S', 'T', 'R', 'I', 'N', 'G'};
    char cpystr1[31] = {'D', 'H', 'R', 'Y', 'S', 'T',        'O', 'N', 'E', ' ', 'P', 'R', 'O', 'G',
                        'R', 'A', 'M', ',', '1', (char)0xde, 'S', 'T', 'S', 'T', 'R', 'I', 'N', 'G'};
    char cpystr2[31] = {'D', 'H', 'R', 'Y', 'S', 'T',        'O', 'N', 'E', ' ', 'P', 'R', 'O', 'G',
                        'R', 'A', 'M', ',', '2', (char)0xde, 'N', 'D', 'S', 'T', 'R', 'I', 'N', 'G'};
    char cpystr3[31] = {'D', 'H', 'R', 'Y', 'S', 'T',        'O', 'N', 'E', ' ', 'P', 'R', 'O', 'G',
                        'R', 'A', 'M', ',', '3', (char)0xde, 'R', 'D', 'S', 'T', 'R', 'I', 'N', 'G'};

    Ptr_Glob->Ptr_Comp = Next_Ptr_Glob;
    Ptr_Glob->Discr = Ident_1;
    Ptr_Glob->variant.var_1.Enum_Comp = Ident_3;
    Ptr_Glob->variant.var_1.Int_Comp = 40;
    strcpy(Ptr_Glob->variant.var_1.Str_Comp, cpystr0);
    strcpy(Str_1_Loc, cpystr1);

    *(Arr_2_Glob + 8 * 50 + 7) = 10;

    Number_Of_Runs = iteration;

#ifdef _DEBUG
    cce::printf("Execution starts, %d runs through Dhrystone\n", Number_Of_Runs);
#endif

    /***************/
    /* Start timer */
    /***************/
    uint64_t tStart = get_sys_cnt();

    for (Run_Index = 1; Run_Index <= Number_Of_Runs; ++Run_Index) {
        Proc_5();
        Proc_4();
        /* Ch_1_Glob == 'A', Ch_2_Glob == 'B', Bool_Glob == true */
        Int_1_Loc = 2;
        Int_2_Loc = 3;
        strcpy(Str_2_Loc, cpystr2);
        Enum_Loc = Ident_2;
        *Bool_Glob = !Func_2(Str_1_Loc, Str_2_Loc);
        /* Bool_Glob == 1 */
        while (Int_1_Loc < Int_2_Loc) /* loop body executed once */
        {
            Int_3_Loc = 5 * Int_1_Loc - Int_2_Loc;
            /* Int_3_Loc == 7 */
            Proc_7(Int_1_Loc, Int_2_Loc, &Int_3_Loc);
            /* Int_3_Loc == 7 */
            Int_1_Loc += 1;
        } /* while */
        /* Int_1_Loc == 3, Int_2_Loc == 3, Int_3_Loc == 7 */
        Proc_8(Arr_1_Glob, Arr_2_Glob, Int_1_Loc, Int_3_Loc);
        /* Int_Glob == 5 */
        Proc_1(Ptr_Glob);
        for (Ch_Index = 'A'; Ch_Index <= *Ch_2_Glob; ++Ch_Index)
        /* loop body executed twice */
        {
            if (Enum_Loc == Func_1(Ch_Index, 'C'))
            /* then, not executed */
            {
                Proc_6(Ident_1, &Enum_Loc);
                strcpy(Str_2_Loc, cpystr3);
                Int_2_Loc = Run_Index;
                *Int_Glob = Run_Index;
            }
        }
        /* Int_1_Loc == 3, Int_2_Loc == 3, Int_3_Loc == 7 */
        Int_2_Loc = Int_2_Loc * Int_1_Loc;
        Int_1_Loc = Int_2_Loc / Int_3_Loc;
        Int_2_Loc = 7 * (Int_2_Loc - Int_3_Loc) - Int_1_Loc;
        /* Int_1_Loc == 1, Int_2_Loc == 13, Int_3_Loc == 7 */
        Proc_2(&Int_1_Loc);
        /* Int_1_Loc == 5 */
    } /* loop "for Run_Index" */

    /**************/
    /* Stop timer */
    /**************/

    pipe_barrier(PIPE_ALL);
    uint64_t tEnd = get_sys_cnt();

#ifdef _DEBUG
    cce::printf("Start @%d End @%d (%d us)\n", int(tStart), int(tEnd), int(tEnd - tStart) * 20 / 1000);
#endif
}

__global__ AICORE __attribute__((aic)) void warmup_kernel()
{}

template <int iteration>
void LaunchTDhrystone(void *stream)
{
    warmup_kernel<<<24, nullptr, stream>>>();
    runTDhrystone<iteration><<<1, nullptr, stream>>>();
}

template void LaunchTDhrystone<1000>(void *stream);
template void LaunchTDhrystone<2000>(void *stream);
template void LaunchTDhrystone<3000>(void *stream);
template void LaunchTDhrystone<4000>(void *stream);
template void LaunchTDhrystone<5000>(void *stream);
template void LaunchTDhrystone<6000>(void *stream);
template void LaunchTDhrystone<7000>(void *stream);
template void LaunchTDhrystone<8000>(void *stream);