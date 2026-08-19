    using LeftTile = TileLeft<float, 16, 16>;
    using RightTile = TileRight<float, 16, 16>;
    using AccTile = TileAcc<float, 16, 16>;
    using BiasTile = Tile<TileType::Bias, float, 1, 16>;
    using LeftScaleTile = TileLeftScale<float, 16, 2>;
    using RightScaleTile = TileRightScale<float, 16, 2>;

    LeftTile lhs;
    RightTile rhs;
    AccTile gemv;
    AccTile gemvAcc;
    AccTile gemvMx;
    AccTile matmulMx;
    AccTile accIn;
    BiasTile bias;
    LeftScaleTile lhsScale;
    RightScaleTile rhsScale;
    size_t addr = 0;
    AssignTileStorage(addr, lhs, rhs, gemv, gemvAcc, gemvMx, matmulMx, accIn, bias, lhsScale, rhsScale);

    FillAll(lhs, 0.0f);
    FillAll(rhs, 0.0f);
    FillAll(accIn, 1.0f);
    FillAll(lhsScale, 2.0f);
    FillAll(rhsScale, 3.0f);
    for (int r = 0; r < lhs.GetValidRow(); ++r) {
        for (int c = 0; c < lhs.GetValidCol(); ++c) {
            SetValue(lhs, r, c, static_cast<float>(r + c + 1));
            SetValue(rhs, r, c, static_cast<float>((r == c) ? 2 : 1));
        }
    }
    for (int c = 0; c < bias.GetValidCol(); ++c) {
        SetValue(bias, 0, c, static_cast<float>(c));
    }

    TGEMV(gemv, lhs, rhs);
    TGEMV_ACC(gemvAcc, accIn, lhs, rhs);
    TGEMV_MX(gemvMx, lhs, lhsScale, rhs, rhsScale);
    TMATMUL_MX(matmulMx, lhs, lhsScale, rhs, rhsScale);

    const auto expectedGemv = ComputeMatmulExpected<AccTile>(lhs, rhs);
    const auto expectedGemvAcc = ComputeMatmulExpected<AccTile>(lhs, rhs, &accIn);
    ExpectTileEqualsVector(gemv, expectedGemv);
    ExpectTileEqualsVector(gemvAcc, expectedGemvAcc);
    ExpectTileEqualsVector(gemvMx, expectedGemv);
    ExpectTileEqualsVector(matmulMx, expectedGemv);

    AccTile gemvBias;
    AssignTileStorage(addr, gemvBias);
    TGEMV_BIAS(gemvBias, lhs, rhs, bias);
    std::vector<float> biasValues(bias.GetValidCol());
    for (int c = 0; c < bias.GetValidCol(); ++c) {
        biasValues[c] = GetValue(bias, 0, c);
    }
    const auto expectedGemvBias = ComputeMatmulExpected<AccTile>(lhs, rhs, nullptr, biasValues.data());
    ExpectTileEqualsVector(gemvBias, expectedGemvBias);
