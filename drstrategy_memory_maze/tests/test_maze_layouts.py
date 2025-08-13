"""Tests for maze layout functionality."""

from __future__ import annotations

import pytest

from drstrategy_memory_maze.maze_layouts import (
    MazeLayout, 
    get_layout, 
    list_layouts, 
    validate_layout,
    LAYOUTS
)


class TestMazeLayout:
    """Test MazeLayout dataclass."""

    def test_maze_layout_immutable(self):
        """Test that MazeLayout is immutable."""
        layout = get_layout('FourRooms7x7')
        
        # Should not be able to modify frozen dataclass
        with pytest.raises(AttributeError):
            layout.max_steps = 1000

    def test_maze_layout_attributes(self):
        """Test MazeLayout has required attributes."""
        layout = get_layout('FourRooms7x7')
        
        assert hasattr(layout, 'layout')
        assert hasattr(layout, 'max_steps')
        assert hasattr(layout, 'len_x')
        assert hasattr(layout, 'len_y')
        assert hasattr(layout, 'rooms')
        assert hasattr(layout, 'invert_origin')
        
        assert isinstance(layout.layout, str)
        assert isinstance(layout.max_steps, int)
        assert isinstance(layout.len_x, int)
        assert isinstance(layout.len_y, int)
        assert isinstance(layout.rooms, list)
        assert callable(layout.invert_origin)

    def test_invert_origin_callable(self):
        """Test invert_origin function works."""
        layout = get_layout('FourRooms7x7')
        
        # Test coordinate transformation
        result = layout.invert_origin([0, 0])
        assert isinstance(result, list)
        assert len(result) == 2


class TestGetLayout:
    """Test get_layout function."""

    def test_get_layout_valid(self):
        """Test getting valid layout."""
        layout = get_layout('FourRooms7x7')
        assert isinstance(layout, MazeLayout)
        assert layout.len_x == 7
        assert layout.len_y == 7
        assert layout.max_steps == 500

    def test_get_layout_all_available(self):
        """Test all available layouts can be retrieved."""
        for layout_name in LAYOUTS.keys():
            layout = get_layout(layout_name)
            assert isinstance(layout, MazeLayout)
            assert layout.layout is not None
            assert layout.max_steps > 0

    def test_get_layout_invalid(self):
        """Test getting invalid layout raises error."""
        with pytest.raises(ValueError, match="Unknown layout"):
            get_layout('NonExistentLayout')

    def test_get_layout_empty_string(self):
        """Test empty string layout name."""
        with pytest.raises(ValueError, match="Unknown layout"):
            get_layout('')

    def test_get_layout_none(self):
        """Test None layout name."""
        with pytest.raises(ValueError, match="Unknown layout"):
            get_layout(None)


class TestListLayouts:
    """Test list_layouts function."""

    def test_list_layouts_returns_list(self):
        """Test list_layouts returns list."""
        layouts = list_layouts()
        assert isinstance(layouts, list)
        assert len(layouts) > 0

    def test_list_layouts_sorted(self):
        """Test returned layouts are sorted."""
        layouts = list_layouts()
        assert layouts == sorted(layouts)

    def test_list_layouts_contains_expected(self):
        """Test list contains expected layout names.""" 
        layouts = list_layouts()
        expected = ['FourRooms7x7', 'FourRooms15x15', 'Maze7x7']
        
        for expected_layout in expected:
            assert expected_layout in layouts

    def test_list_layouts_all_strings(self):
        """Test all layout names are strings."""
        layouts = list_layouts()
        assert all(isinstance(name, str) for name in layouts)
        assert all(name.strip() for name in layouts)  # No empty strings


class TestValidateLayout:
    """Test validate_layout function."""

    def test_validate_layout_valid(self):
        """Test validation of valid layout strings."""
        valid_layout = \"\"\"\n*****\n*P G*\n*****\n\"\"\"
        assert validate_layout(valid_layout) is True

    def test_validate_layout_from_constants(self):
        """Test validation of layout constants."""
        for layout_data in LAYOUTS.values():
            assert validate_layout(layout_data.layout) is True

    def test_validate_layout_empty(self):
        """Test validation of empty layout."""
        assert validate_layout('') is False

    def test_validate_layout_none(self):
        """Test validation of None layout."""
        assert validate_layout(None) is False

    def test_validate_layout_not_string(self):
        """Test validation of non-string layout."""
        assert validate_layout(123) is False
        assert validate_layout(['*', '*']) is False

    def test_validate_layout_too_small(self):
        """Test validation of too small layout."""
        too_small = \"*\\n*\"  # Only 2 lines
        assert validate_layout(too_small) is False

    def test_validate_layout_inconsistent_width(self):
        """Test validation of layout with inconsistent line widths."""
        inconsistent = \"\"\"\n*****\n*P*\n*****\n\"\"\"  # Middle line shorter
        assert validate_layout(inconsistent) is False


class TestLayoutConstants:
    """Test layout string constants."""

    def test_layouts_dict_not_empty(self):
        """Test LAYOUTS dictionary is not empty."""
        assert len(LAYOUTS) > 0

    def test_all_layouts_valid(self):
        """Test all layout constants are valid."""
        for name, layout_data in LAYOUTS.items():
            assert isinstance(name, str)
            assert isinstance(layout_data, MazeLayout)
            assert validate_layout(layout_data.layout) is True

    def test_layout_names_descriptive(self):
        """Test layout names are descriptive."""
        for name in LAYOUTS.keys():
            # Names should indicate room structure and size
            assert any(indicator in name for indicator in ['Rooms', 'Maze'])
            assert any(size in name for size in ['7x7', '15x15', '30x30'])

    def test_layout_sizes_match_names(self):
        """Test layout sizes match their names."""
        layout_7x7 = get_layout('FourRooms7x7')
        assert layout_7x7.len_x == 7
        assert layout_7x7.len_y == 7
        
        layout_15x15 = get_layout('FourRooms15x15')
        assert layout_15x15.len_x == 15
        assert layout_15x15.len_y == 15

    def test_max_steps_reasonable(self):
        """Test max_steps are reasonable for maze sizes."""
        for layout_data in LAYOUTS.values():
            # Larger mazes should generally have more steps
            size_factor = layout_data.len_x * layout_data.len_y
            assert layout_data.max_steps >= size_factor  # At least area
            assert layout_data.max_steps <= size_factor * 10  # Not too excessive