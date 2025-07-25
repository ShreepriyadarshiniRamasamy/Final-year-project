import PropTypes from 'prop-types';
import { isLabelInValue } from './util';
var internalValProp = PropTypes.oneOfType([PropTypes.string, PropTypes.number]);
export function genArrProps(propType) {
  return PropTypes.oneOfType([propType, PropTypes.arrayOf(propType)]);
}
/**
 * Origin code check `multiple` is true when `treeCheckStrictly` & `labelInValue`.
 * But in process logic is already cover to array.
 * Check array is not necessary. Let's simplify this check logic.
 */

export function valueProp() {
  for (var _len = arguments.length, args = new Array(_len), _key = 0; _key < _len; _key++) {
    args[_key] = arguments[_key];
  }

  var props = args[0],
      propName = args[1],
      Component = args[2];

  if (isLabelInValue(props)) {
    var _err = genArrProps(PropTypes.shape({
      label: PropTypes.node,
      value: internalValProp
    })).apply(void 0, args);

    if (_err) {
      return new Error("Invalid prop `".concat(propName, "` supplied to `").concat(Component, "`. ") + "You should use { label: string, value: string | number } or [{ label: string, value: string | number }] instead.");
    }

    return null;
  }

  var err = genArrProps(internalValProp).apply(void 0, args);

  if (err) {
    return new Error("Invalid prop `".concat(propName, "` supplied to `").concat(Component, "`. ") + "You should use string or [string] instead.");
  }

  return null;
}