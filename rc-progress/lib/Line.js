'use strict';

exports.__esModule = true;

var _extends2 = require('babel-runtime/helpers/extends');

var _extends3 = _interopRequireDefault(_extends2);

var _objectWithoutProperties2 = require('babel-runtime/helpers/objectWithoutProperties');

var _objectWithoutProperties3 = _interopRequireDefault(_objectWithoutProperties2);

var _classCallCheck2 = require('babel-runtime/helpers/classCallCheck');

var _classCallCheck3 = _interopRequireDefault(_classCallCheck2);

var _possibleConstructorReturn2 = require('babel-runtime/helpers/possibleConstructorReturn');

var _possibleConstructorReturn3 = _interopRequireDefault(_possibleConstructorReturn2);

var _inherits2 = require('babel-runtime/helpers/inherits');

var _inherits3 = _interopRequireDefault(_inherits2);

var _react = require('react');

var _react2 = _interopRequireDefault(_react);

var _enhancer = require('./enhancer');

var _enhancer2 = _interopRequireDefault(_enhancer);

var _types = require('./types');

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { 'default': obj }; }

var Line = function (_Component) {
  (0, _inherits3['default'])(Line, _Component);

  function Line() {
    var _temp, _this, _ret;

    (0, _classCallCheck3['default'])(this, Line);

    for (var _len = arguments.length, args = Array(_len), _key = 0; _key < _len; _key++) {
      args[_key] = arguments[_key];
    }

    return _ret = (_temp = (_this = (0, _possibleConstructorReturn3['default'])(this, _Component.call.apply(_Component, [this].concat(args))), _this), _this.paths = {}, _temp), (0, _possibleConstructorReturn3['default'])(_this, _ret);
  }

  Line.prototype.render = function render() {
    var _this2 = this;

    var _props = this.props,
        className = _props.className,
        percent = _props.percent,
        prefixCls = _props.prefixCls,
        strokeColor = _props.strokeColor,
        strokeLinecap = _props.strokeLinecap,
        strokeWidth = _props.strokeWidth,
        style = _props.style,
        trailColor = _props.trailColor,
        trailWidth = _props.trailWidth,
        restProps = (0, _objectWithoutProperties3['default'])(_props, ['className', 'percent', 'prefixCls', 'strokeColor', 'strokeLinecap', 'strokeWidth', 'style', 'trailColor', 'trailWidth']);


    delete restProps.gapPosition;

    var percentList = Array.isArray(percent) ? percent : [percent];
    var strokeColorList = Array.isArray(strokeColor) ? strokeColor : [strokeColor];

    var center = strokeWidth / 2;
    var right = 100 - strokeWidth / 2;
    var pathString = 'M ' + (strokeLinecap === 'round' ? center : 0) + ',' + center + '\n           L ' + (strokeLinecap === 'round' ? right : 100) + ',' + center;
    var viewBoxString = '0 0 100 ' + strokeWidth;

    var stackPtg = 0;

    return _react2['default'].createElement(
      'svg',
      (0, _extends3['default'])({
        className: prefixCls + '-line ' + className,
        viewBox: viewBoxString,
        preserveAspectRatio: 'none',
        style: style
      }, restProps),
      _react2['default'].createElement('path', {
        className: prefixCls + '-line-trail',
        d: pathString,
        strokeLinecap: strokeLinecap,
        stroke: trailColor,
        strokeWidth: trailWidth || strokeWidth,
        fillOpacity: '0'
      }),
      percentList.map(function (ptg, index) {
        var pathStyle = {
          strokeDasharray: ptg + 'px, 100px',
          strokeDashoffset: '-' + stackPtg + 'px',
          transition: 'stroke-dashoffset 0.3s ease 0s, stroke-dasharray .3s ease 0s, stroke 0.3s linear'
        };
        var color = strokeColorList[index] || strokeColorList[strokeColorList.length - 1];

        stackPtg += ptg;

        return _react2['default'].createElement('path', {
          key: index,
          className: prefixCls + '-line-path',
          d: pathString,
          strokeLinecap: strokeLinecap,
          stroke: color,
          strokeWidth: strokeWidth,
          fillOpacity: '0',
          ref: function ref(path) {
            _this2.paths[index] = path;
          },
          style: pathStyle
        });
      })
    );
  };

  return Line;
}(_react.Component);

Line.propTypes = _types.propTypes;

Line.defaultProps = _types.defaultProps;

exports['default'] = (0, _enhancer2['default'])(Line);
module.exports = exports['default'];