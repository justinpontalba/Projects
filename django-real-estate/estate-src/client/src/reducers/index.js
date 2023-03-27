import { combineReducers } from "redux";
import { propertiesListReducer } from './propertyreducers'

export default combineReducers({
    propertiesList: propertiesListReducer,
});