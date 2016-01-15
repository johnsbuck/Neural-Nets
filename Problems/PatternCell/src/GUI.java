import javax.swing.*;
import javax.swing.border.Border;
import javax.swing.border.MatteBorder;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.io.*;
import java.util.List;
import java.util.ArrayList;

/**
 * Basic Cell Grid GUI used for printing out numeric boolean (0 & 1) values for each selected
 * and non-selected cell.
 */
public class GUI
{
  private GridPane grid;
  private JFrame frame;
  private JButton setButton;
  private JTextField fileName;

  /**
   * Main function for running GUI
   * @param args
   */
  public static void main(String[] args)
  {
    new GUI();
  }

  /**
   * Default Constructor
   */
  public GUI()
  {
    grid = new GridPane();

    frame = new JFrame("Grid");
    frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    frame.setLayout(new BorderLayout());

    setButton = new JButton("What number is this?");
    setButton.setPreferredSize(new Dimension(160, 40));
    setButton.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e)
      {
        try {
          PrintWriter out = new PrintWriter(fileName.getText());
          out.println(grid.toString());
          out.close();
        } catch (FileNotFoundException er) {
          er.printStackTrace();
        }
        grid.clear();
      }
    });

    fileName = new JTextField("File name");

    frame.add(grid, BorderLayout.CENTER);
    frame.add(setButton, BorderLayout.PAGE_END);
    frame.add(fileName, BorderLayout.PAGE_START);
    frame.pack();
    frame.setLocationRelativeTo(null);
    frame.setVisible(true);
  }

  /**
   * Grid of 2 dimensional cells.
   *
   * Default size: 7 x 5
   */
  private class GridPane extends JPanel
  {
    private int colCount;
    private int rowCount;
    private List<CellPane> cells = new ArrayList<>(colCount * rowCount);

    /**
     * Default Constructor
     */
    public GridPane()
    {
      colCount = 5;
      rowCount = 7;

      init();
    }

    /**
     * Custom Grid Constructor
     *
     * Sets the number of rows and columns by the user.
     * If number of rows <= 0 then will default to 8.
     * If number of columns <= 0 then will default to 8.
     *
     * @param row
     * @param col
     */
    public GridPane(int row, int col)
    {
      rowCount = row > 0 ? row : 8;
      colCount = col > 0 ? col : 8;

      init();
    }

    /**
     * Initializes the pane by adding cells
     */
    private void init()
    {
      setLayout(new GridBagLayout());

      GridBagConstraints gbc = new GridBagConstraints();

      for(int row = 0; row < rowCount; row++)
      {
        for(int col = 0; col < colCount; col++)
        {
          gbc.gridx = col;
          gbc.gridy = row;

          cells.add(new CellPane());
          CellPane cellPane = cells.get(cells.size() - 1);

          Border border = null;
          if(row < rowCount - 1)
          {
            if(col < colCount - 1)
            {
              border = new MatteBorder(1, 1, 0, 0, Color.GRAY);
            }
            else
            {
              border = new MatteBorder(1, 1, 0, 1, Color.GRAY);
            }
          }
          else
          {
            if(col < colCount - 1)
            {
              border = new MatteBorder(1, 1, 1, 0, Color.GRAY);
            }
            else
            {
              border = new MatteBorder(1, 1, 1, 1, Color.GRAY);
            }
          }

          cellPane.setBorder(border);
          add(cellPane, gbc);
        }
      }
    }

    /**
     * Sets all cells to unselected.
     */
    public void clear()
    {
      for(int i = 0; i < cells.size(); i++)
      {
        cells.get(i).clear();
      }
    }

    /**
     * Returns a list of cells.
     *
     * @return
     */
    public List<CellPane> getCells()
    {
      return cells;
    }

    /**
     * Returns a list of integers where 1s represents selected cells
     * and 0 represents unselected cells.
     *
     * @return
     */
    public List<Integer> getValues()
    {
      List<Integer> cellValues = new ArrayList(cells.size());

      for(int i = 0; i < cells.size(); i++)
      {
        cellValues.add(cells.get(i).getValue());
      }

      return cellValues;
    }

    /**
     * Returns a string set of cell values.
     *
     * @return
     */
    public String toString() {
      String str = "";
      List<Integer> cellValues = this.getValues();

      for(int i = 0; i < cellValues.size(); i++)
      {
        str += cellValues.get(i);
        if(i < cellValues.size() - 1)
        {
          str += " ";
        }
      }

      return str;
    }
  }

  /**
   * Individual cell from grid.
   * They may be selected or unselected by clicking and returns 1 when selected and 0 when not.
   */
  public class CellPane extends JPanel
  {
    private Color defaultBackground;

    /**
     * Default Constructor
     */
    public CellPane()
    {
      setPreferredSize(new Dimension(50, 50));
      defaultBackground = getBackground();

      addMouseListener(new MouseAdapter()
      {
        @Override
        public void mouseClicked(MouseEvent e)
        {
          if (getBackground().getBlue() == 0)
          {
            setBackground(defaultBackground);
          }
          else
          {
            setBackground(Color.BLACK);
          }
        }
      });
    }

    /**
     * Sets cell to unselected
     */
    public void clear()
    {
      setBackground(defaultBackground);
    }

    /**
     * Returns 1 when cell is selected and 0 when it is not selected.
     *
     * @return
     */
    public int getValue()
    {
      if(getBackground().getBlue() == 0) {
        return 1;
      }

      return 0;
    }

    /**
     * Returns cell's value as a string.
     *
     * @return
     */
    public String toString()
    {
      return Integer.toString(this.getValue());
    }
  }
}