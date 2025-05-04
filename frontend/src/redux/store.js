import { configureStore } from "@reduxjs/toolkit";
import userReducer from "../redux/user/userSlice";
import tableReducer from "../redux/table/tableSlice";
import imagesReducer, { addID } from "../redux/images/imagesSlice";


const store = configureStore({
  reducer: {
    user: userReducer,
    table: tableReducer,
    images: imagesReducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware().concat(addID), 
});

export default store;
