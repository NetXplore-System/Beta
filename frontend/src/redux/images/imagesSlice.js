import { createSlice } from "@reduxjs/toolkit";
import toast from "react-hot-toast";
import { v4 as uuidv4 } from 'uuid';

const initialState = {
    main: [],
    comparison: {
        source: [],
        comparison: [],
    },
};

const imagesSlice = createSlice({
    name: "images",
    initialState,
    reducers: {
        addToMain(state, action) {
            state.main.push({ ...action.payload, type: 'main' });
            toast.success("Image added to main list.");
        },
        addToComparison(state, action) {
            state.comparison[action.payload.type].push({ ...action.payload, type: action.payload.type });
            toast.success("Image added to comparison list.");
        },
        replaseImage(state, action) {
            toast.success("Image already exists for this graph, replacing it.");
            // state.images[action.payload.imageIndex] = action.payload;
        },
        clearImages(state) {
            state.main = [];
            state.comparison.comparison = [];
            state.comparison.source = [];
        },
    },
});

export const { addToMain, addToComparison, clearImages } = imagesSlice.actions;
export const addID = (store) => (next) => (action) => {
    if (action.type === "images/addToMain" || action.type === "images/addToComparison") {
        action.payload.id = uuidv4();
        return next(action);
    }

    return next(action);
};

export default imagesSlice.reducer;
