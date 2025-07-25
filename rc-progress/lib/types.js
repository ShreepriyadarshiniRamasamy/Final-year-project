'use strict';

exports.__esModule = true;
exports.propTypes = exports.defaultProps = undefined;

var _propTypes = require('prop-types');

var _propTypes2 = _interopRequireDefault(_propTypes);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { 'default': obj }; }

var defaultProps = exports.defaultProps = {
  className: '',
  percent: 0,
  prefixCls: 'rc-progress',
  strokeColor: '#2db7f5',
  strokeLinecap: 'round',
  strokeWidth: 1,
  style: {},
  trailColor: '#D9D9D9',
  trailWidth: 1
};

var mixedType = _propTypes2['default'].oneOfType([_propTypes2['default'].number, _propTypes2['default'].string]);

var propTypes = exports.propTypes = {
  className: _propTypes2['default'].string,
  percent: _propTypes2['default'].oneOfType([mixedType, _propTypes2['default'].arrayOf(mixedType)]),
  prefixCls: _propTypes2['default'].string,
  strokeColor: _propTypes2['default'].oneOfType([_propTypes2['default'].string, _propTypes2['default'].arrayOf(_propTypes2['default'].string)]),
  strokeLinecap: _propTypes2['default'].oneOf(['butt', 'round', 'square']),
  strokeWidth: mixedType,
  style: _propTypes2['default'].object,
  trailColor: _propTypes2['default'].string,
  trailWidth: mixedType
};